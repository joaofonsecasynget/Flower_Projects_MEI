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
        
        # Retorna os parâmetros atualizados, o tamanho do conjunto de dados e as métricas
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
        
        # Gerar explicações após avaliação final
        try:
            logger.info("Generating LIME explanation...")
            lime_exp = explainer.explain_lime(X_train, X_test[0])
            lime_path = base_reports / "lime_explanation.txt"
            with open(lime_path, "w") as f:
                f.write(str(lime_exp.as_list()))
            logger.info(f"LIME explanation saved to {lime_path}")
            
            logger.info("Generating SHAP explanation...")
            shap_values = explainer.explain_shap(X_train, n_samples=50)
            shap_path = base_reports / "shap_values.npy"
            np.save(shap_path, shap_values)
            logger.info(f"SHAP values saved to {shap_path}")
        except Exception as e:
            logger.error(f"Error generating explanations: {str(e)}")
        
        # Guardar info.txt e métricas
        with open(base_reports / "info.txt", "w") as f:
            f.write(f"Client {args.cid} - Num samples: {len(client_data)}\n")
            f.write(f"Test loss: {test_loss:.4f}\nTest RMSE: {test_rmse:.2f}\n")
        
        # Retorna a perda, tamanho do conjunto de dados e métricas
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
