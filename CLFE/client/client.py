import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import time
from utils import LinearClassificationModel, CustomDataset, train, evaluate
from explainability import ModelExplainer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import logging
import flwr as fl
import json
import matplotlib.pyplot as plt
import sys
from report_utils import generate_evolution_plots, generate_html_report, save_artifacts, generate_explainability

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

# Limpar diretórios de relatórios e arquivos de histórico para evitar dados residuais
def clean_previous_reports(directory):
    """Limpa arquivos de relatórios anteriores para evitar histórico acumulado."""
    if not directory.exists():
        return
    
    logger.info(f"Limpando diretório de relatórios anteriores: {directory}")
    
    # Lista de padrões de arquivo para limpar
    patterns_to_clean = [
        "*.png",  # Todos os gráficos
        "*.html",  # Relatórios HTML
        "*.json",  # Arquivos JSON (incluindo histórico de métricas)
        "*.txt",   # Arquivos de texto
        "*.npy"    # Arrays numpy salvos
    ]
    
    for pattern in patterns_to_clean:
        for file_path in directory.glob(pattern):
            if file_path.is_file():
                file_path.unlink()
                logger.info(f"Removido arquivo: {file_path}")

# Limpar relatórios anteriores para evitar acumular histórico
clean_previous_reports(base_reports)

# Carregar dataset completo
logger.info(f"Loading dataset from {args.dataset_path}")
df = pd.read_csv(args.dataset_path)

# --- Validação e limpeza de dados ---
# 1. Remover colunas constantes (zero variância)
# Temporariamente comentado conforme solicitado
# const_cols = [col for col in df.columns if df[col].nunique() == 1]
# if const_cols:
#     logger.info(f"Removing constant columns: {const_cols}")
#     df = df.drop(columns=const_cols)

# 2. Remover colunas de identificação e tempo (não preditivas)
irrelevant_cols = ['índice', 'indice', 'imeisv']  # Removido '_time' para permitir extração de features
cols_to_remove = [col for col in irrelevant_cols if col in df.columns]
if cols_to_remove:
    logger.info(f"Removing identifier/time columns: {cols_to_remove}")
    df = df.drop(columns=cols_to_remove)

# 2.1. Extração de features temporais a partir de _time (se existir)
if '_time' in df.columns:
    logger.info("Extracting temporal features from _time")
    try:
        # Tentar conversão padrão primeiro
        df['_time'] = pd.to_datetime(df['_time'])
    except ValueError:
        # Se falhar, tentar com format='ISO8601' para lidar com diferentes formatos ISO
        logger.info("Standard datetime parsing failed, trying with ISO8601 format")
        df['_time'] = pd.to_datetime(df['_time'], format='ISO8601')
    
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

# Definir a coluna target
target_col = "attack"  # Definir explicitamente o nome da coluna target
if target_col not in df.columns:
    logger.error(f"Target column '{target_col}' not found in dataset! Available columns: {df.columns.tolist()}")
    sys.exit(1)

# Log da distribuição global do target
logger.info(f"Global target distribution: {df[target_col].value_counts().to_dict()}")
logger.info(f"Total dataset size: {len(df)}")

# Particionamento estratificado para preservar distribuição do target
# Primeiro, dividimos em treino/teste global para garantir consistência
skf = StratifiedKFold(n_splits=args.num_total_clients, shuffle=True, random_state=args.seed)
cliente_atual = 0
for idx_train, idx_test in skf.split(df, df[target_col]):
    cliente_atual += 1
    if cliente_atual == args.cid:
        client_data = df.iloc[idx_test].copy()
        break

logger.info(f"Client data shape: {client_data.shape}")

# Separar features e target
X = client_data.drop(columns=[target_col])
y = client_data[target_col]

# Log das features e target para verificação
logger.info(f"Target column: {target_col}")
logger.info(f"Target values distribution: {y.value_counts().to_dict()}")
logger.info(f"Number of features: {X.shape[1]}")
logger.info(f"Features list: {X.columns.tolist()}")

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
model = LinearClassificationModel(input_size).to(device)
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
        
        # Criar diretórios para armazenar resultados
        self.base_reports = Path("reports") / f"client_{args.cid}"
        self.base_results = Path("results") / f"client_{args.cid}"
        self.base_reports.mkdir(parents=True, exist_ok=True)
        self.base_results.mkdir(parents=True, exist_ok=True)
        
        # Inicializar o dicionário de histórico para métricas
        self.reset_history()
        
        # Track do tempo para fit/evaluate
        self.fit_start_time = 0
        self.eval_start_time = 0
        
        # Feature names para explicabilidade
        self.feature_names = []
        if 'X' in globals() and isinstance(X, pd.DataFrame):
            self.feature_names = list(X.columns)
            
    def reset_history(self):
        """Reinicializa o histórico de métricas."""
        logger.info("Reinicializando histórico de métricas")
        self.history = {
            # Métricas de treino
            "train_loss": [],
            "fit_duration_seconds": [],
            
            # Métricas de validação
            "val_loss": [],
            "val_rmse": [],
            "val_accuracy": [],
            "val_precision": [],
            "val_recall": [],
            "val_f1": [],
            
            # Métricas de teste
            "test_loss": [],
            "test_rmse": [],
            "test_accuracy": [],
            "test_precision": [],
            "test_recall": [],
            "test_f1": [],
            "evaluate_duration_seconds": [],
            
            # Métricas de explicabilidade
            "lime_duration_seconds": [],
            "shap_duration_seconds": []
        }
        
    # Retorna os parâmetros do modelo como uma lista de arrays numpy.
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    # Atualiza os parâmetros do modelo com os recebidos do servidor.
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
    
    # Treina o modelo nos dados locais.
    def fit(self, parameters, config):
        """Treina o modelo nos dados locais."""
        # Reinicia o histórico se for a primeira ronda
        if config.get('round_number', 0) == 1:
            self.reset_history()
            # Também limpa os relatórios para garantir dados limpos
            clean_previous_reports(self.base_reports)
        
        start_time = time.time()
        logger.info(f"[CID: {args.cid}] Received training instructions: {config}")
        self.set_parameters(parameters)
        train_loss = train(self.model, self.train_loader, epochs=self.epochs, device=self.device)
        val_loss, val_rmse, _, _, val_accuracy, val_precision, val_recall, val_f1 = evaluate(self.model, self.val_loader, device=self.device)
        logger.info(f"[CID: {args.cid}] Round {config['round_number']} - Train loss: {train_loss:.4f}")
        logger.info(f"[CID: {args.cid}] Round {config['round_number']} - Val loss: {val_loss:.4f} | Val RMSE: {val_rmse:.2f}")
        logger.info(f"[CID: {args.cid}] Round {config['round_number']} - Val metrics: Acc: {val_accuracy:.4f}, Prec: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")
        self.history["train_loss"].append(train_loss)
        self.history["val_accuracy"].append(val_accuracy)
        self.history["val_precision"].append(val_precision)
        self.history["val_recall"].append(val_recall)
        self.history["val_f1"].append(val_f1)
        end_time = time.time()
        duration = end_time - start_time
        logger.info(f"[CID: {args.cid}] Round {config['round_number']} - Fit duration: {duration:.2f} seconds")
        self.history["fit_duration_seconds"].append(duration)

        # Retornar parâmetros atualizados, número de exemplos e métricas (incluindo duração)
        return self.get_parameters(config={}), len(self.train_loader.dataset), {
            "train_loss": float(train_loss),
            "val_accuracy": float(val_accuracy),
            "val_precision": float(val_precision),
            "val_recall": float(val_recall),
            "val_f1": float(val_f1),
            "fit_duration_seconds": float(duration)
        }

    def evaluate(self, parameters, config):
        """Avalia o modelo nos dados locais e gera métricas/relatório."""
        self.set_parameters(parameters)
        
        # Verificar se estamos na última ronda e se faltam métricas de treinamento
        is_final_round = config.get('round_number', 0) == config.get('num_rounds', 0)
        if is_final_round and len(self.history["train_loss"]) < config.get('round_number', 0):
            logger.info(f"[CID: {args.cid}] Última ronda: fazendo fit adicional para garantir métricas completas")
            # Realizar treinamento adicional para obter métricas na última ronda
            start_time = time.time()
            train_loss = train(self.model, self.train_loader, epochs=self.epochs, device=self.device)
            end_time = time.time()
            duration = end_time - start_time
            
            # Registrar métricas
            self.history["train_loss"].append(train_loss)
            self.history["fit_duration_seconds"].append(duration)
            logger.info(f"[CID: {args.cid}] Round {config['round_number']} - Garantindo registro do Train loss: {train_loss:.4f}")
            logger.info(f"[CID: {args.cid}] Round {config['round_number']} - Garantindo registro do Fit duration: {duration:.2f} seconds")
        
        # Medir tempo total da avaliação
        start_time = time.time()
        
        # Calcular métricas no conjunto de validação
        val_loss, val_rmse, _, _, val_accuracy, val_precision, val_recall, val_f1 = evaluate(self.model, self.val_loader, self.device)
        
        # Calcular duração da avaliação
        pure_eval_duration = time.time() - start_time
        
        # Adicionar métricas ao histórico
        self.history["val_loss"].append(val_loss)
        self.history["val_rmse"].append(val_rmse)
        
        # Verificar se estas métricas já não foram adicionadas nesta ronda (para prevenir duplicação)
        current_round = config.get('round_number', 0)
        if len(self.history["val_accuracy"]) < current_round:
            self.history["val_accuracy"].append(val_accuracy)
            self.history["val_precision"].append(val_precision)
            self.history["val_recall"].append(val_recall)
            self.history["val_f1"].append(val_f1)
        
        self.history["evaluate_duration_seconds"].append(pure_eval_duration)
        
        # Calcular métricas de teste
        test_loss, test_rmse, _, _, test_accuracy, test_precision, test_recall, test_f1 = evaluate(self.model, self.test_loader, self.device)
        
        # Adicionar métricas de teste ao histórico
        self.history["test_loss"].append(float(test_loss))
        self.history["test_rmse"].append(float(test_rmse))
        
        # Verificar se estas métricas já não foram adicionadas nesta ronda (para prevenir duplicação)
        if len(self.history["test_accuracy"]) < current_round:
            self.history["test_accuracy"].append(test_accuracy)
            self.history["test_precision"].append(test_precision)
            self.history["test_recall"].append(test_recall)
            self.history["test_f1"].append(test_f1)
        
        # Inicializar durações para LIME/SHAP
        lime_duration = 0.0
        shap_duration = 0.0
        
        # --- Explicabilidade LIME/SHAP ---
        # Executar apenas na última ronda para economizar recursos
        if config['round_number'] == config['num_rounds']:
            # Primeiro, vamos salvar X_train para gerar explicabilidade
            try:
                X_train_batch = next(iter(self.train_loader))[0].numpy()
                X_train_path = self.base_reports / "X_train.npy"
                np.save(X_train_path, X_train_batch)
                logger.info(f"Saved X_train for explainability at {X_train_path}")
                
                # Agora gerar explicabilidade
                explainer = ModelExplainer(self.model, feature_names=list(X.columns), device=self.device)
                lime_duration, shap_duration = generate_explainability(explainer, X_train_path, self.base_reports)
            except Exception as e:
                logger.error(f"Error during explainability generation: {e}", exc_info=True)
        
        # Adicionar durações ao histórico
        self.history["lime_duration_seconds"].append(lime_duration)
        self.history["shap_duration_seconds"].append(shap_duration)
        
        # --- Geração de Gráficos e Relatório ---
        # Gerar apenas se já tivermos dados (pelo menos uma ronda completa)
        if self.history["train_loss"]:
            try:
                # Gerar gráficos de evolução
                plot_files = generate_evolution_plots(self.history, self.base_reports)
                
                # Salvar histórico de métricas como JSON
                history_path = self.base_reports / "metrics_history.json"
                with open(history_path, "w") as f:
                    json.dump(self.history, f, indent=4, default=lambda x: float(x) if isinstance(x, np.float32) else x)
                logger.info(f"Metrics history saved to {history_path}")
                
                # Gerar relatório HTML final
                generate_html_report(
                    self.history, 
                    plot_files, 
                    self.base_reports, 
                    args.cid, 
                    args.dataset_path, 
                    args.seed, 
                    args.epochs, 
                    args.server_address
                )
                
                # Salvar artefatos (info.txt e modelo)
                save_artifacts(
                    self.base_reports, 
                    self.base_results, 
                    self.model, 
                    args.cid, 
                    args.dataset_path, 
                    args.seed, 
                    args.epochs, 
                    args.server_address
                )
            except Exception as e:
                logger.error(f"Error generating plots or report: {e}", exc_info=True)
        
        # Retornar métricas de teste para agregação do servidor
        return float(test_loss), len(self.test_loader.dataset), {
            "test_loss": float(test_loss),
            "test_rmse": float(test_rmse),
            "test_accuracy": float(test_accuracy),
            "test_precision": float(test_precision),
            "test_recall": float(test_recall),
            "test_f1": float(test_f1),
        }

# Iniciar a sessão de aprendizado federado
if __name__ == "__main__":
    # Criando a instância do cliente
    client = RLFEClient(model, train_loader, val_loader, test_loader, args.epochs, device)
    
    # Iniciar a conexão com o servidor Flower
    logger.info(f"Starting Flower client, connecting to server at {args.server_address}")
    fl.client.start_numpy_client(server_address=args.server_address, client=client)
    
    logger.info("Federated learning finished.")
