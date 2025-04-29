import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import time
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
            "test_rmse": [],
            "fit_duration_seconds": [],
            "evaluate_duration_seconds": [],
            "lime_duration_seconds": [],
            "shap_duration_seconds": [],
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
        start_time = time.time()
        logger.info(f"[CID: {args.cid}] Received training instructions: {config}")
        self.set_parameters(parameters)
        train_loss = train(self.model, self.train_loader, epochs=self.epochs, device=self.device)
        val_loss, val_rmse, _, _ = evaluate(self.model, self.val_loader, device=self.device)
        logger.info(f"[CID: {args.cid}] Round {config['round_number']} - Train loss: {train_loss:.4f}")
        logger.info(f"[CID: {args.cid}] Round {config['round_number']} - Val loss: {val_loss:.4f} | Val RMSE: {val_rmse:.2f}")
        self.history["train_loss"].append(train_loss)
        self.history["val_loss"].append(val_loss)
        self.history["val_rmse"].append(val_rmse)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"[CID: {args.cid}] Round {config['round_number']} - Fit duration: {duration:.2f} seconds")
        self.history["fit_duration_seconds"].append(duration)

        # Retornar parâmetros atualizados, número de exemplos e métricas (incluindo duração)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_rmse": float(val_rmse),
            "fit_duration_seconds": float(duration)
        }

    def evaluate(self, parameters, config):
        logger.info(f"[CID: {args.cid}] Received evaluation instructions: {config}")
        self.set_parameters(parameters)

        # Medir apenas o tempo da avaliação no test_loader
        start_pure_eval = time.perf_counter() # Use perf_counter for higher resolution
        test_loss, test_rmse, y_pred, y_true = evaluate(self.model, self.test_loader, device=self.device)
        end_pure_eval = time.perf_counter() # Use perf_counter for higher resolution
        pure_eval_duration = end_pure_eval - start_pure_eval

        logger.info(f"[CID: {args.cid}] Round {config['round_number']} - Test loss: {test_loss:.4f} | Test RMSE: {test_rmse:.2f} | Pure evaluation duration: {pure_eval_duration:.4f}s") # Log with higher precision initially
        self.history["test_loss"].append(test_loss)
        self.history["test_rmse"].append(test_rmse)
        self.history["evaluate_duration_seconds"].append(pure_eval_duration) # Guardar tempo da avaliação pura

        # Geração das explicações LIME e SHAP (só na última ronda)
        lime_duration = 0.0 # Inicializar fora do bloco
        shap_duration = 0.0 # Inicializar fora do bloco
        try:
            X_train_path = base_reports / "X_train.npy"
            # Executar apenas se X_train existir E for a última ronda
            # if X_train_path.exists(): # Condição anterior (executava sempre se X_train existisse)
            if X_train_path.exists() and config['round_number'] == config['num_rounds']:
                logger.info(f"[CID: {args.cid}] Running LIME/SHAP explanations for the final round ({config['round_number']})...")
                # Garantir que o diretório base para relatórios existe
                base_reports.mkdir(exist_ok=True, parents=True)
                X_train_local = np.load(X_train_path)
                
                # LIME
                start_lime = time.time()
                # Usar uma instância aleatória ou a primeira para LIME
                instance_index = np.random.randint(0, len(X_train_local))
                instance = X_train_local[instance_index]
                # Passar num_features explicitamente - CORREÇÃO: Remover num_features daqui, já é tratado internamente --> Revertendo para o original
                lime_exp = explainer.explain_lime(X_train_local, instance)
                lime_exp.save_to_file(str(base_reports / "lime_final.html"))
                lime_exp_fig = lime_exp.as_pyplot_figure()
                
                # Adicionar '(Top 10 Features)' ao título e ajustar layout
                ax = lime_exp_fig.gca() # Obter eixos
                current_title = ax.get_title()
                ax.set_title(current_title + " (Top 10 Features)")
                lime_exp_fig.tight_layout() # Ajustar layout para melhor legibilidade
                
                lime_exp_fig.savefig(base_reports / "lime_final.png")
                plt.close(lime_exp_fig) # Fechar a figura para libertar memória
                with open(base_reports / "lime_explanation.txt", "w") as f:
                    f.write(str(lime_exp.as_list()))
                end_lime = time.time()
                lime_duration = end_lime - start_lime # Atualizar a duração
                logger.info(f"[CID: {args.cid}] LIME duration: {lime_duration:.2f} seconds")

                # SHAP
                start_shap = time.time()
                shap_values = explainer.explain_shap(X_train_local, n_samples=100) # n_samples pode ser ajustado
                import shap as shap_lib
                # Gerar o plot (sem show=False para que ele fique ativo no matplotlib)
                shap_lib.summary_plot(shap_values, X_train_local, feature_names=explainer.feature_names)
                # Obter a figura atual e salvá-la
                fig = plt.gcf()
                fig.savefig(base_reports / "shap_final.png", bbox_inches='tight') # bbox_inches='tight' ajuda a evitar cortes
                plt.close(fig) # Fechar a figura específica
                np.save(base_reports / "shap_values.npy", shap_values)
                end_shap = time.time()
                shap_duration = end_shap - start_shap # Atualizar a duração
                logger.info(f"[CID: {args.cid}] SHAP duration: {shap_duration:.2f} seconds")
            
            # else: # Se não for a última ronda ou X_train não existir
            #     logger.info(f"[CID: {args.cid}] Skipping LIME/SHAP for round {config['round_number']}.")
                
        except ImportError as ie:
            logger.error(f"Import error during LIME/SHAP generation: {str(ie)}. Make sure LIME and SHAP are installed.")
            # Manter durações como 0.0 em caso de erro
        except Exception as e:
            logger.error(f"Error during LIME/SHAP generation: {str(e)}")
            # Manter durações como 0.0 em caso de erro na última ronda
            # lime_duration = 0.0 # Já inicializado
            # shap_duration = 0.0 # Já inicializado
            
        # Adicionar durações ao histórico (serão 0.0 exceto na última ronda bem-sucedida)
        # Nota: Se preferir que as listas no histórico SÓ tenham o valor da última ronda de explicabilidade, 
        # mova estas duas linhas para dentro do bloco `if ... and round_number == num_rounds:`
        self.history["lime_duration_seconds"].append(lime_duration)
        self.history["shap_duration_seconds"].append(shap_duration)
        
        # Adicionar a duração da avaliação PURA (sem LIME/SHAP) ao histórico - JÁ FEITO ACIMA
        # self.history["evaluate_duration_seconds"].append(eval_duration)

        # --- Geração de Gráficos de Evolução e Relatório Final --- 
        # Gerar gráficos e relatório apenas na última ronda para evitar sobrecarga
        plot_files = []
        metrics_to_plot = ["train_loss", "val_loss", "test_loss", "val_rmse", "test_rmse"]
        if len(self.history["train_loss"]) > 0:
            rounds = list(range(1, len(self.history["train_loss"]) + 1))
            for metric_key in metrics_to_plot:
                if metric_key in self.history and self.history[metric_key] and any(v is not None and not np.isnan(v) for v in self.history[metric_key]):
                    try:
                        plt.figure(figsize=(10, 5))
                        values = self.history[metric_key]
                        valid_rounds = [r for r, v in zip(rounds, values) if v is not None and not np.isnan(v)]
                        valid_values = [v for v in values if v is not None and not np.isnan(v)]

                        if valid_rounds:
                            plt.plot(valid_rounds, valid_values, marker='o', linestyle='-', label=metric_key)
                            plt.xlabel("Ronda")
                            plt.ylabel(metric_key.replace('_', ' ').title())
                            plt.title(f"Evolução do {metric_key.replace('_', ' ').title()}")
                            plt.legend()
                            plt.xticks(rounds)
                            plt.grid(True, axis='y', linestyle='--')

                            plot_filename = f"{metric_key}_evolution.png"
                            plot_path = base_reports / plot_filename
                            plt.savefig(plot_path)
                            plt.close()
                            plot_files.append(plot_filename)
                            logger.info(f"Generated plot: {plot_path}")
                        else:
                            logger.warning(f"Skipping plot for '{metric_key}' as no valid (non-NaN/None) data points found.")
                    except Exception as e:
                        logger.error(f"Error generating plot for {metric_key}: {e}", exc_info=True)
                else:
                    logger.warning(f"Metric key '{metric_key}' not found in history, is empty, or contains only NaN/None, skipping plot.")
        else:
            logger.warning("No rounds completed, skipping plot generation.")

        # Gerar relatório HTML final
        from datetime import datetime
        dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_content = f"""
        <!DOCTYPE html>
        <html lang="pt">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Relatório Federado Detalhado - Cliente {args.cid}</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f4f7f6; color: #333; }}
                h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #1abc9c; padding-bottom: 5px;}}
                h1 {{ text-align: center; margin-bottom: 30px; }}
                img {{ max-width: 800px; height: auto; margin: 15px auto; display: block; border: 1px solid #ddd; border-radius: 4px; padding: 5px; background-color: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
                .metrics-table {{ border-collapse: collapse; margin: 25px 0; width: 100%; box-shadow: 0 2px 3px rgba(0,0,0,0.1); background-color: white; font-size: 0.9em; }}
                .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
                .metrics-table th {{ background-color: #1abc9c; color: white; text-transform: capitalize; font-weight: 600; position: sticky; top: 0; z-index: 1; }}
                .metrics-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metrics-table tr:hover {{ background-color: #f1f1f1; }}
                .metrics-table td:first-child, .metrics-table th:first-child {{ text-align: center; font-weight: bold; }}
                .metrics-table td {{ min-width: 80px; }}
                .table-container {{ max-height: 500px; overflow-y: auto; border: 1px solid #ddd; margin-bottom: 20px; }}
                ul {{ list-style-type: square; padding-left: 20px; margin-top: 10px; }}
                li {{ margin-bottom: 5px; }}
                li a {{ color: #3498db; text-decoration: none; }}
                li a:hover {{ text-decoration: underline; }}
                p em {{ color: #7f8c8d; font-size: 0.9em; text-align: center; display: block; margin-top: 20px; }}
                .section {{ background-color: white; padding: 20px; margin-bottom: 20px; border-radius: 5px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
                .plot-container img {{ margin-bottom: 25px; }}
                /* Estilos para a tabela de plots */
                .plot-table {{ width: 100%; border-collapse: separate; border-spacing: 15px; }}
                .plot-table td {{ vertical-align: top; text-align: center; width: 33%; background-color: #fff; padding: 10px; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); border: 1px solid #eee; }}
                .plot-table img {{ max-width: 100%; height: auto; margin: 10px auto; display: block; }}
                .plot-table h3 {{ margin-top: 0; font-size: 1em; border-bottom: 1px solid #eee; padding-bottom: 5px; margin-bottom: 10px; color: #555; }}
            </style>
        </head>
        <body>
            <h1>Relatório Federado Detalhado - Cliente {args.cid}</h1>

            <div class="section">
                <h2>Evolução Detalhada das Métricas e Tempos por Ronda</h2>
        """
        # Gerar Tabela HTML Dinamicamente
        num_rounds_completed = len(self.history["train_loss"])
        if num_rounds_completed > 0 and self.history:
            html_content += '<div class="table-container">\n<table class="metrics-table">\n<thead>\n<tr><th>Ronda</th>'
            all_keys = list(self.history.keys()) # Usar as chaves existentes no histórico
            # Ordenar chaves para melhor leitura (opcional, mas recomendado)
            key_order = ["train_loss", "val_loss", "test_loss", "val_rmse", "test_rmse",
                         "fit_duration_seconds", "evaluate_duration_seconds",
                         "lime_duration_seconds", "shap_duration_seconds"]
            ordered_keys = [k for k in key_order if k in all_keys] + [k for k in all_keys if k not in key_order]

            for key in ordered_keys:
                html_content += f"<th>{key.replace('_', ' ').title()}</th>"
            html_content += "</tr>\n</thead>\n<tbody>\n"

            for r in range(num_rounds_completed):
                html_content += f"<tr><td>{r + 1}</td>"
                for key in ordered_keys: # Iterar com a ordem definida
                    value = self.history.get(key, [None]*num_rounds_completed)[r] # Acesso seguro
                    # Formatar: 4 casas decimais para métricas, 4 para tempos. Tratar NaN/None.
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        formatted_value = "N/A"
                    elif isinstance(value, (int, float)):
                        if 'duration' in key or 'seconds' in key:
                            formatted_value = f"{value:.4f}" # Changed to 4 decimal places
                        else:
                            formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    html_content += f"<td>{formatted_value}</td>"
                html_content += "</tr>\n"
            html_content += "</tbody>\n</table>\n</div>"
        else:
            html_content += "<p><em>Não há dados históricos para apresentar na tabela.</em></p>"
        html_content += "</div>" # Fim da section da tabela


        # Incluir Plots de Evolução Gerados
        html_content += '<div class="section plot-container">\n<h2>Plots de Evolução das Métricas</h2>\n'
        if plot_files:
            html_content += '<table class="plot-table">\n'
            num_plots = len(plot_files)
            plots_per_row = 3
            for i, plot_file in enumerate(plot_files):
                if i % plots_per_row == 0:
                    html_content += '<tr>\n'
                
                plot_title = plot_file.replace('_evolution.png','').replace('_', ' ').title()
                html_content += f'<td><h3>{plot_title}</h3><img src="{plot_file}" alt="Evolução {plot_title}" /></td>\n'
                
                if (i + 1) % plots_per_row == 0 or (i + 1) == num_plots:
                    # Se for o último plot da linha OU o último plot de todos
                    # Adicionar células vazias se a linha não estiver completa
                    if (i + 1) == num_plots and (i + 1) % plots_per_row != 0:
                        remaining_cells = plots_per_row - ((i + 1) % plots_per_row)
                        for _ in range(remaining_cells):
                            html_content += '<td></td>\n' # Célula vazia
                    html_content += '</tr>\n'
            html_content += '</table>\n'
        else:
             html_content += "<p><em>Nenhum gráfico de evolução foi gerado ou encontrado.</em></p>\n"
        html_content += "</div>" # Fim da section dos plots

 
        # Secção de Explicabilidade (mantém-se, usa os ficheiros finais gerados antes)
        html_content += f"""
            <div class="section">
                <h2>Explicabilidade (Resultados Finais da Última Ronda)</h2>
                <h3>LIME</h3>
                {f'<img src="lime_final.png" alt="LIME final" style="max-width: 600px;"/>' if 'lime_final.png' in os.listdir(base_reports) else '<p><em>Explicação LIME não disponível ou não gerada.</em></p>'}
                <p><a href="lime_final.html" target="_blank">Ver LIME interativo</a> | <a href="lime_explanation.txt" target="_blank">Ver LIME (texto)</a></p>
                <h3>SHAP</h3>
                {f'<img src="shap_final.png" alt="SHAP final" style="max-width: 600px;"/>' if 'shap_final.png' in os.listdir(base_reports) else '<p><em>Explicação SHAP não disponível ou não gerada.</em></p>'}
                <p><a href="shap_values.npy" download>Download SHAP Values (.npy)</a></p>
            </div>
        """

        # Secção de Artefactos (mantém-se)
        html_content += f"""
            <div class="section">
                <h2>Artefactos e Downloads</h2>
                <ul>
                    <li><a href="metrics_history.json" target="_blank">Histórico Completo (Métricas e Tempos - JSON)</a></li>
                    <li><a href="../results/client_{args.cid}/model_client_{args.cid}.pt" download>Modelo Treinado Final (.pt)</a></li>
                    <li><a href="info.txt" target="_blank">Informações Gerais (info.txt)</a></li>
                    {f'<li><a href="lime_explanation.txt" target="_blank">Explicação LIME (texto)</a></li>' if 'lime_explanation.txt' in os.listdir(base_reports) else ''}
                    {f'<li><a href="lime_final.html" target="_blank">Explicação LIME (interativo HTML)</a></li>' if 'lime_final.html' in os.listdir(base_reports) else ''}
                    {f'<li><a href="shap_values.npy" download>Valores SHAP (.npy)</a></li>' if 'shap_values.npy' in os.listdir(base_reports) else ''}
                </ul>
            </div>

            <p><em>Relatório gerado automaticamente em {dt_str}</em></p>
        </body>
        </html>
        """
        # Salvar o HTML final
        try:
            report_path = base_reports / "final_report.html"
            with open(report_path, "w", encoding='utf-8') as f:
                f.write(html_content)
            logger.info(f"Final HTML report generated at {report_path}")
        except Exception as e:
            logger.error(f"Error saving final HTML report: {e}", exc_info=True)

        # Guardar info.txt e modelo final (código existente mantido)
        try:
            info_path = base_reports / "info.txt"
            with open(info_path, "w") as f:
                f.write(f"Client ID: {args.cid}\n")
                f.write(f"Dataset Path: {args.dataset_path}\n")
                f.write(f"Seed: {args.seed}\n")
                f.write(f"Epochs: {args.epochs}\n")
                f.write(f"Server Address: {args.server_address}\n")
            logger.info(f"Info file saved to {info_path}")

            model_path = base_results / f"model_client_{args.cid}.pt"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Final model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving info.txt or final model: {e}", exc_info=True)

        # Retornar métricas de teste para agregação do servidor
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
