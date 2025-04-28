import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
from utils import LinearRegressionModel, CustomDataset, train, evaluate
from explainability import ModelExplainer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging
import flwr as fl
import json
import matplotlib.pyplot as plt

# Filtro para garantir que todas as mensagens de log tenham o campo 'cid'
class AddClientIdFilter(logging.Filter):
    def filter(self, record):
        if not hasattr(record, 'cid'):
            record.cid = '?'
        return True

# Adaptação: garantir que logs de bibliotecas externas também tenham 'cid' usando LoggerAdapter
class ClientLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        if 'extra' not in kwargs:
            kwargs['extra'] = {}
        if 'cid' not in kwargs['extra']:
            kwargs['extra']['cid'] = args.cid if 'args' in globals() else '?'
        return msg, kwargs

logger = logging.getLogger()
logger.addFilter(AddClientIdFilter())
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = ClientLoggerAdapter(logger, {'cid': args.cid if 'args' in globals() else '?'})

# Argumentos do script
parser = argparse.ArgumentParser(description="RLFE Federated Client")
parser.add_argument('--cid', type=int, required=True, help='ID do cliente (1-indexed)')
parser.add_argument('--num_clients', type=int, required=True, help='Número de clientes a executar neste teste')
parser.add_argument('--num_total_clients', type=int, default=10, help='Número total de clientes para particionamento dos dados')
parser.add_argument('--dataset_path', type=str, default="../DatasetIOT/transformed_dataset_imeisv_8642840401612300.csv", help="Path to dataset CSV")
parser.add_argument('--seed', type=int, default=42, help='Seed para reprodutibilidade')
parser.add_argument('--epochs', type=int, default=5, help="Epochs per round")
parser.add_argument('--server_address', type=str, default="server:9091", help="Endereço do servidor Flower")
args = parser.parse_args()

# Garantir reprodutibilidade
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Criar pastas de output para este cliente
base_reports = Path("reports") / f"client_{args.cid}"
base_results = Path("results") / f"client_{args.cid}"
base_reports.mkdir(parents=True, exist_ok=True)
base_results.mkdir(parents=True, exist_ok=True)

# Carregar dataset completo
logger.info(f"Loading dataset from {args.dataset_path}")
df = pd.read_csv(args.dataset_path)

# --- Validação e limpeza de dados ---
# 1. Remover colunas constantes (zero variância)
const_cols = [col for col in df.columns if df[col].nunique() == 1]
if const_cols:
    logger.info(f"Removing constant columns: {const_cols}")
    df = df.drop(columns=const_cols)

# 2. Remover colunas de identificação e tempo (não preditivas)
irrelevant_cols = ['índice', '_time', 'imeisv']
cols_to_remove = [col for col in irrelevant_cols if col in df.columns]
if cols_to_remove:
    logger.info(f"Removing identifier/time columns: {cols_to_remove}")
    df = df.drop(columns=cols_to_remove)

# 2.1. Extração de features temporais a partir de _time (se existir)
if '_time' in df.columns:
    logger.info("Extracting temporal features from _time")
    df['_time'] = pd.to_datetime(df['_time'])
    df['hour'] = df['_time'].dt.hour
    df['dayofweek'] = df['_time'].dt.dayofweek
    df['day'] = df['_time'].dt.day
    df['month'] = df['_time'].dt.month
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    # Remover _time após extração
    df = df.drop(columns=['_time'])

# 3. Tratamento de valores em falta (missing values)
# Substituir valores em falta por mediana (mais robusto que média)
missing_cols = df.columns[df.isnull().any()].tolist()
if missing_cols:
    logger.info(f"Filling missing values in columns: {missing_cols}")
    for col in missing_cols:
        mediana = df[col].median()
        df[col] = df[col].fillna(mediana)

# 4. Codificação de variáveis categóricas (one-hot encoding)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_cols:
    logger.info(f"Applying one-hot encoding to columns: {categorical_cols}")
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
# --- Fim da validação e limpeza ---

# Particionamento on-the-fly
# O dataset é sempre dividido por num_total_clients, independentemente do número de clientes executados
partes = np.array_split(df, args.num_total_clients)
client_data = partes[args.cid - 1]  # Numeração começa em 1
logger.info(f"Client data shape: {client_data.shape}")

# Separar features e target
X = client_data.drop(columns=[client_data.columns[-1]])
y = client_data[client_data.columns[-1]]

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=args.seed)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=args.seed)

# Salvar X_train para explicabilidade LIME/SHAP
np.save(base_reports / "X_train.npy", X_train)

train_ds = CustomDataset(X_train, y_train)
val_ds = CustomDataset(X_val, y_val)
test_ds = CustomDataset(X_test, y_test)
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)

# Modelo, explicador e device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_size = X_train.shape[1]
model = LinearRegressionModel(input_size).to(device)
explainer = ModelExplainer(model, feature_names=list(X.columns), device=device)

# Definindo a classe do cliente Flower
class RLFEClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader, epochs, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device

        # Histórico de métricas por ronda
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "val_rmse": [],
            "test_loss": [],
            "test_rmse": []
        }

    def get_parameters(self, config):
        """Retorna os parâmetros do modelo como uma lista de arrays numpy."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Atualiza os parâmetros do modelo com os recebidos do servidor."""
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        """Treina o modelo nos dados locais."""
        self.set_parameters(parameters)
        train_loss = train(self.model, self.train_loader, epochs=self.epochs, device=self.device)
        val_loss, val_rmse, _, _ = evaluate(self.model, self.val_loader, device=self.device)
        logger.info(f"Client {args.cid} - Train loss: {train_loss:.4f}")
        logger.info(f"Client {args.cid} - Val loss: {val_loss:.4f} | Val RMSE: {val_rmse:.2f}")
        self.history["train_loss"].append(float(train_loss))
        self.history["val_loss"].append(float(val_loss))
        self.history["val_rmse"].append(float(val_rmse))
        # Guardar histórico incremental
        hist_path = base_reports / "metrics_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)

        # Salvar X_train para explicabilidade (se for a última ronda)
        round_number = config.get("round_number", -1)
        num_rounds = config.get("num_rounds", -1)
        if round_number == num_rounds:
            try:
                # Salvar X_train (features de treino) para explicabilidade
                np.save(base_reports / "X_train.npy", X_train)
            except Exception as e:
                logger.error(f"Erro ao salvar X_train.npy: {e}")

            # Geração das explicações LIME e SHAP
            try:
                X_train_path = base_reports / "X_train.npy"
                if X_train_path.exists():
                    X_train_local = np.load(X_train_path)
                    # LIME
                    instance = X_train_local[0]
                    lime_exp = explainer.explain_lime(X_train_local, instance)
                    lime_exp.save_to_file(str(base_reports / "lime_final.html"))
                    lime_exp_fig = lime_exp.as_pyplot_figure()
                    lime_exp_fig.savefig(base_reports / "lime_final.png")
                    with open(base_reports / "lime_explanation.txt", "w") as f:
                        f.write(str(lime_exp.as_list()))
                    lime_ok = True
                else:
                    lime_ok = False
                # SHAP
                try:
                    shap_values = explainer.explain_shap(X_train_local, n_samples=100)
                    import shap as shap_lib
                    shap_lib.summary_plot(shap_values, X_train_local, feature_names=explainer.feature_names, show=False)
                    import matplotlib.pyplot as plt
                    plt.savefig(base_reports / "shap_final.png")
                    np.save(base_reports / "shap_values.npy", shap_values)
                    shap_ok = True
                except Exception as e:
                    with open(base_reports / "shap_error.txt", "w") as f:
                        f.write(str(e))
                    shap_ok = False
            except Exception as e:
                with open(base_reports / "lime_shap_error.txt", "w") as f:
                    f.write(str(e))
                lime_ok = False
                shap_ok = False

            # Relatório HTML
            from datetime import datetime
            dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(base_reports / "final_report.html", "w") as f:
                f.write(f"""
            <html>
            <head>
                <title>Relatório Federado - Cliente {args.cid}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1, h2 {{ color: #2c3e50; }}
                    img {{ max-width: 600px; margin: 20px 0; }}
                    .metrics-table {{ border-collapse: collapse; margin: 20px 0; }}
                    .metrics-table th, .metrics-table td {{ border: 1px solid #ccc; padding: 8px; }}
                    .metrics-table th {{ background: #f0f0f0; }}
                </style>
            </head>
            <body>
                <h1>Relatório Federado - Cliente {args.cid}</h1>
                <h2>Resumo das Métricas</h2>
                <table class="metrics-table">
                    <tr><th>Métrica</th><th>Valor Final</th></tr>
                    <tr><td>Train Loss</td><td>{self.history['train_loss'][-1] if self.history['train_loss'] else 'N/A'}</td></tr>
                    <tr><td>Val Loss</td><td>{self.history['val_loss'][-1] if self.history['val_loss'] else 'N/A'}</td></tr>
                    <tr><td>Val RMSE</td><td>{self.history['val_rmse'][-1] if self.history['val_rmse'] else 'N/A'}</td></tr>
                    <tr><td>Test Loss</td><td>{self.history['test_loss'][-1] if self.history['test_loss'] else 'N/A'}</td></tr>
                    <tr><td>Test RMSE</td><td>{self.history['test_rmse'][-1] if self.history['test_rmse'] else 'N/A'}</td></tr>
                </table>
                <h2>Plots de Evolução</h2>
                <img src="loss_evolution.png" alt="Evolução do Loss" />
                <img src="rmse_evolution.png" alt="Evolução do RMSE" />
                <h2>Explicabilidade</h2>
                <h3>LIME (última ronda)</h3>
                {f'<img src="lime_final.png" alt="LIME final" />' if lime_ok else '<p><em>Explicabilidade LIME não disponível.</em></p>'}
                <h3>SHAP (última ronda)</h3>
                {f'<img src="shap_final.png" alt="SHAP final" />' if shap_ok else '<p><em>Explicabilidade SHAP não disponível.</em></p>'}
                <h2>Artefactos e Downloads</h2>
                <ul>
                    <li><a href="metrics_history.json">Histórico de Métricas (JSON)</a></li>
                    <li><a href="../results/client_{args.cid}/model_client_{args.cid}.pt">Modelo Treinado (.pt)</a></li>
                    <li><a href="info.txt">Info.txt</a></li>
                    <li><a href="lime_explanation.txt">LIME Explanation (texto)</a></li>
                    <li><a href="shap_values.npy">SHAP Values (.npy)</a></li>
                </ul>
                <p><em>Gerado automaticamente em {dt_str}</em></p>
            </body>
            </html>
            """)
            # Plots de evolução
            try:
                rounds = list(range(1, len(self.history["train_loss"]) + 1))
                plt.figure()
                plt.plot(rounds, self.history["train_loss"], label="Train Loss")
                plt.plot(rounds, self.history["val_loss"], label="Val Loss")
                plt.plot(rounds, self.history["test_loss"], label="Test Loss")
                plt.xlabel("Ronda")
                plt.ylabel("Loss")
                plt.title("Evolução do Loss")
                plt.legend()
                plt.xticks(rounds)  # Apenas inteiros
                plt.savefig(base_reports / "loss_evolution.png")
                plt.close()
                plt.figure()
                plt.plot(rounds, self.history["val_rmse"], label="Val RMSE")
                plt.plot(rounds, self.history["test_rmse"], label="Test RMSE")
                plt.xlabel("Ronda")
                plt.ylabel("RMSE")
                plt.title("Evolução do RMSE")
                plt.legend()
                plt.xticks(rounds)  # Apenas inteiros
                plt.savefig(base_reports / "rmse_evolution.png")
                plt.close()
                logger.info(f"Metric plots saved to {base_reports}")
            except Exception as e:
                logger.error(f"Error generating/saving metric plots: {str(e)}")
            # Guardar info.txt
            with open(base_reports / "info.txt", "w") as f:
                if self.history['test_loss'] and self.history['test_rmse']:
                    f.write(f"Client {args.cid} - Num samples: {len(self.test_loader.dataset)}\n")
                    f.write(f"Test loss: {self.history['test_loss'][-1]:.4f}\nTest RMSE: {self.history['test_rmse'][-1]:.2f}\n")
                else:
                    f.write("Client {args.cid} - Num samples: {len(self.test_loader.dataset)}\n")
                    f.write("Test loss: N/A\nTest RMSE: N/A\n")
            # Guardar modelo treinado final
            try:
                import torch
                model_path = base_results / f"model_client_{args.cid}.pt"
                torch.save(self.model.state_dict(), model_path)
                logger.info(f"Final model saved to {model_path}")
            except Exception as e:
                logger.error(f"Error saving model: {str(e)}")
        return self.get_parameters({}), len(self.train_loader.dataset), {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_rmse": float(val_rmse)
        }

    def evaluate(self, parameters, config):
        """Avalia o modelo nos dados de teste locais."""
        self.set_parameters(parameters)
        test_loss, test_rmse, y_pred, y_true = evaluate(self.model, self.test_loader, device=self.device)
        logger.info(f"Client {args.cid} - Test loss: {test_loss:.4f} | Test RMSE: {test_rmse:.2f}")
        self.history["test_loss"].append(float(test_loss))
        self.history["test_rmse"].append(float(test_rmse))
        # Guardar histórico incremental
        hist_path = base_reports / "metrics_history.json"
        with open(hist_path, "w") as f:
            json.dump(self.history, f, indent=2)
        # Só gerar outputs finais se for a última ronda
        round_number = config.get("round_number", -1)
        num_rounds = config.get("num_rounds", -1)
        if round_number == num_rounds:
            # Geração das explicações LIME e SHAP
            try:
                X_train_path = base_reports / "X_train.npy"
                if X_train_path.exists():
                    X_train_local = np.load(X_train_path)
                    # LIME
                    instance = X_train_local[0]
                    lime_exp = explainer.explain_lime(X_train_local, instance)
                    lime_exp.save_to_file(str(base_reports / "lime_final.html"))
                    lime_exp_fig = lime_exp.as_pyplot_figure()
                    lime_exp_fig.savefig(base_reports / "lime_final.png")
                    with open(base_reports / "lime_explanation.txt", "w") as f:
                        f.write(str(lime_exp.as_list()))
                    lime_ok = True
                else:
                    lime_ok = False
                # SHAP
                try:
                    shap_values = explainer.explain_shap(X_train_local, n_samples=100)
                    import shap as shap_lib
                    shap_lib.summary_plot(shap_values, X_train_local, feature_names=explainer.feature_names, show=False)
                    import matplotlib.pyplot as plt
                    plt.savefig(base_reports / "shap_final.png")
                    np.save(base_reports / "shap_values.npy", shap_values)
                    shap_ok = True
                except Exception as e:
                    with open(base_reports / "shap_error.txt", "w") as f:
                        f.write(str(e))
                    shap_ok = False
            except Exception as e:
                with open(base_reports / "lime_shap_error.txt", "w") as f:
                    f.write(str(e))
                lime_ok = False
                shap_ok = False

            # Relatório HTML
            from datetime import datetime
            dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(base_reports / "final_report.html", "w") as f:
                f.write(f"""
            <html>
            <head>
                <title>Relatório Federado - Cliente {args.cid}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; }}
                    h1, h2 {{ color: #2c3e50; }}
                    img {{ max-width: 600px; margin: 20px 0; }}
                    .metrics-table {{ border-collapse: collapse; margin: 20px 0; }}
                    .metrics-table th, .metrics-table td {{ border: 1px solid #ccc; padding: 8px; }}
                    .metrics-table th {{ background: #f0f0f0; }}
                </style>
            </head>
            <body>
                <h1>Relatório Federado - Cliente {args.cid}</h1>
                <h2>Resumo das Métricas</h2>
                <table class="metrics-table">
                    <tr><th>Métrica</th><th>Valor Final</th></tr>
                    <tr><td>Train Loss</td><td>{self.history['train_loss'][-1] if self.history['train_loss'] else 'N/A'}</td></tr>
                    <tr><td>Val Loss</td><td>{self.history['val_loss'][-1] if self.history['val_loss'] else 'N/A'}</td></tr>
                    <tr><td>Val RMSE</td><td>{self.history['val_rmse'][-1] if self.history['val_rmse'] else 'N/A'}</td></tr>
                    <tr><td>Test Loss</td><td>{self.history['test_loss'][-1] if self.history['test_loss'] else 'N/A'}</td></tr>
                    <tr><td>Test RMSE</td><td>{self.history['test_rmse'][-1] if self.history['test_rmse'] else 'N/A'}</td></tr>
                </table>
                <h2>Plots de Evolução</h2>
                <img src="loss_evolution.png" alt="Evolução do Loss" />
                <img src="rmse_evolution.png" alt="Evolução do RMSE" />
                <h2>Explicabilidade</h2>
                <h3>LIME (última ronda)</h3>
                {f'<img src="lime_final.png" alt="LIME final" />' if lime_ok else '<p><em>Explicabilidade LIME não disponível.</em></p>'}
                <h3>SHAP (última ronda)</h3>
                {f'<img src="shap_final.png" alt="SHAP final" />' if shap_ok else '<p><em>Explicabilidade SHAP não disponível.</em></p>'}
                <h2>Artefactos e Downloads</h2>
                <ul>
                    <li><a href="metrics_history.json">Histórico de Métricas (JSON)</a></li>
                    <li><a href="../results/client_{args.cid}/model_client_{args.cid}.pt">Modelo Treinado (.pt)</a></li>
                    <li><a href="info.txt">Info.txt</a></li>
                    <li><a href="lime_explanation.txt">LIME Explanation (texto)</a></li>
                    <li><a href="shap_values.npy">SHAP Values (.npy)</a></li>
                </ul>
                <p><em>Gerado automaticamente em {dt_str}</em></p>
            </body>
            </html>
            """)
            # Plots de evolução
            try:
                rounds = list(range(1, len(self.history["train_loss"]) + 1))
                plt.figure()
                plt.plot(rounds, self.history["train_loss"], label="Train Loss")
                plt.plot(rounds, self.history["val_loss"], label="Val Loss")
                plt.plot(rounds, self.history["test_loss"], label="Test Loss")
                plt.xlabel("Ronda")
                plt.ylabel("Loss")
                plt.title("Evolução do Loss")
                plt.legend()
                plt.xticks(rounds)  # Apenas inteiros
                plt.savefig(base_reports / "loss_evolution.png")
                plt.close()
                plt.figure()
                plt.plot(rounds, self.history["val_rmse"], label="Val RMSE")
                plt.plot(rounds, self.history["test_rmse"], label="Test RMSE")
                plt.xlabel("Ronda")
                plt.ylabel("RMSE")
                plt.title("Evolução do RMSE")
                plt.legend()
                plt.xticks(rounds)  # Apenas inteiros
                plt.savefig(base_reports / "rmse_evolution.png")
                plt.close()
                logger.info(f"Metric plots saved to {base_reports}")
            except Exception as e:
                logger.error(f"Error generating/saving metric plots: {str(e)}")
            # Guardar info.txt
            with open(base_reports / "info.txt", "w") as f:
                if self.history['test_loss'] and self.history['test_rmse']:
                    f.write(f"Client {args.cid} - Num samples: {len(self.test_loader.dataset)}\n")
                    f.write(f"Test loss: {self.history['test_loss'][-1]:.4f}\nTest RMSE: {self.history['test_rmse'][-1]:.2f}\n")
                else:
                    f.write("Client {args.cid} - Num samples: {len(self.test_loader.dataset)}\n")
                    f.write("Test loss: N/A\nTest RMSE: N/A\n")
            # Guardar modelo treinado final
            try:
                import torch
                model_path = base_results / f"model_client_{args.cid}.pt"
                torch.save(self.model.state_dict(), model_path)
                logger.info(f"Final model saved to {model_path}")
            except Exception as e:
                logger.error(f"Error saving model: {str(e)}")
        return float(test_loss), len(self.test_loader.dataset), {
            "test_loss": float(test_loss),
            "test_rmse": float(test_rmse)
        }

# Iniciar a sessão de aprendizado federado
if __name__ == "__main__":
    # Criando a instância do cliente
    client = RLFEClient(model, train_loader, val_loader, test_loader, args.epochs, device)
    
    # Iniciar a conexão com o servidor Flower
    logger.info(f"Starting Flower client, connecting to server at {args.server_address}")
    fl.client.start_numpy_client(server_address=args.server_address, client=client)
    
    logger.info("Federated learning finished.")
