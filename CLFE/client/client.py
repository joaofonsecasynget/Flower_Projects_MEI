
import random
import argparse
import os
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import time
from client.utils import LinearClassificationModel, CustomDataset, train, evaluate
from .explainability import ModelExplainer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
import logging
import flwr as fl
import json
import matplotlib.pyplot as plt
import sys
from .report_helpers.plot_generation import generate_evolution_plots
from .report_helpers.html_generation import generate_html_report
from .report_helpers.artifact_saving import save_artifacts
from .report_helpers.explainability_processing import generate_explainability
from .report_helpers.explainability_processing import generate_explainability_formatado

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
parser = argparse.ArgumentParser(description="CLFE Federated Client")
parser.add_argument('--cid', type=int, required=True, help='ID do cliente (1-indexed)')
parser.add_argument('--num_clients', type=int, required=True, help='Número de clientes a executar neste teste')
parser.add_argument('--num_total_clients', type=int, default=10, help='Número total de clientes para particionamento dos dados')
parser.add_argument('--dataset_path', type=str, default="DatasetIOT/transformed_dataset_imeisv_8642840401612300.csv", help="Path to dataset CSV")
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
irrelevant_cols = ['indice', 'imeisv']  # Removido '_time' para permitir extração de features
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
feature_names = list(X.columns) # Capturar nomes das features
logger.info(f"Features list: {feature_names}")

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train/val/test
X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=args.seed, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=args.seed, stratify=y_temp)

# Criar DataFrame para X_train escalado com nomes de colunas
X_train_df_scaled = pd.DataFrame(X_train, columns=feature_names)

# Calcular pos_weight para BCEWithLogitsLoss
# Assumindo que y_train é uma Pandas Series e contém 0 para 'normal' e 1 para 'ataque'
num_normal = (y_train == 0).sum()
num_ataque = (y_train == 1).sum()

if num_ataque == 0: # Evitar divisão por zero se não houver amostras de ataque em y_train para este cliente
    logger.warning(f"Cliente {args.cid}: Nenhuma amostra de ataque encontrada em y_train. Usando pos_weight=1.0. Distribuição y_train: {y_train.value_counts().to_dict()}")
    pos_weight_value = 1.0
else:
    pos_weight_value = float(num_normal) / float(num_ataque)
logger.info(f"Cliente {args.cid}: Calculado pos_weight para a função de perda: {pos_weight_value:.4f} (Normal: {num_normal}, Ataque: {num_ataque})")

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
explainer = ModelExplainer(model, feature_names=feature_names, device=device) # Usar a variável feature_names

# Definindo a classe do cliente Flower
class CLFEClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, test_loader, epochs, device, pos_weight,
                 X_train_df_scaled, feature_names, scaler): # NOVOS PARÂMETROS
        self.cid = args.cid # Capturar o CID dos argumentos globais
        logger.info(f"CLIENT {self.cid}: CLFEClient.__init__ - START")
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.device = device
            # Instanciar a função de perda com o pos_weight

        self.pos_weight = pos_weight # Armazenar pos_weight
        self.criterion = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([self.pos_weight], device=self.device))


        logger.info(f"CLIENT {self.cid}: Train loader has {len(self.train_loader.dataset)} samples.")
        logger.info(f"CLIENT {self.cid}: Validation loader has {len(self.val_loader.dataset)} samples.")
        logger.info(f"CLIENT {self.cid}: Test loader has {len(self.test_loader.dataset)} samples.")
    # Instanciar a função de perda com o pos_weight
    
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
    
    # Feature names, DataFrame de treino escalado e scaler para explicabilidade e potencial desnormalização
        self.X_train_df_scaled = X_train_df_scaled
        self.feature_names = feature_names
        self.scaler = scaler
        # self.X_train_df_scaled é atribuído, vamos usá-lo diretamente para LIME se for DataFrame
        if self.X_train_df_scaled is not None and hasattr(self.X_train_df_scaled, 'shape'):
            logger.info(f"CLIENT {self.cid}: X_train_df_scaled (background for LIME) has shape: {self.X_train_df_scaled.shape}")
        elif self.X_train_df_scaled is not None: # Caso não seja um DataFrame mas exista
            logger.info(f"CLIENT {self.cid}: X_train_df_scaled (background for LIME) exists, type: {type(self.X_train_df_scaled)}")
        else:
            logger.warning(f"CLIENT {self.cid}: X_train_df_scaled (background for LIME) is None.")
        logger.info(f"CLIENT {self.cid}: CLFEClient.__init__ - END (after all assignments)")
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
        logger.info(f"CLIENT {self.cid}: CLFEClient.get_parameters - START")
        params = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        logger.info(f"CLIENT {self.cid}: CLFEClient.get_parameters - END")
        return params
    
    # Atualiza os parâmetros do modelo com os recebidos do servidor.
    def set_parameters(self, parameters):
        logger.info(f"CLIENT {self.cid}: CLFEClient.set_parameters - START")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)
        logger.info(f"CLIENT {self.cid}: CLFEClient.set_parameters - END (after model.load_state_dict)")
    
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
        train_loss = train(self.model, self.train_loader, epochs=self.epochs, criterion=self.criterion, device=self.device)
        val_loss, val_rmse, _, _, val_accuracy, val_precision, val_recall, val_f1 = evaluate(self.model, self.val_loader, criterion=self.criterion, device=self.device)
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
        logger.info(f"CLIENT {self.cid}: CLFEClient.evaluate - START")
        """Avalia o modelo nos dados locais e gera métricas/relatório."""
        self.set_parameters(parameters)
        
        server_round = config.get('round_number')
        if server_round is None:
            # Fallback para 'server_round' se 'round_number' não existir, embora 'round_number' seja o esperado.
            server_round = config.get('server_round')
            if server_round is None:
                logger.warning(f"CLIENT {self.cid}: 'round_number' nem 'server_round' encontrados no config para evaluate. A assumir ronda 0.")
                server_round = 0 # Ou outra forma de lidar com o erro, se necessário
            else:
                logger.info(f"CLIENT {self.cid}: A utilizar 'server_round' do config: {server_round} para a variável server_round.")
        else:
            logger.info(f"CLIENT {self.cid}: A utilizar 'round_number' do config: {server_round} para a variável server_round.")
        
        # Verificar se estamos na última ronda e se faltam métricas de treinamento
        is_final_round = config.get('round_number', 0) == config.get('num_rounds', 0)
        if is_final_round and len(self.history["train_loss"]) < config.get('round_number', 0):
            logger.info(f"[CID: {args.cid}] Última ronda: fazendo fit adicional para garantir métricas completas")
            # Realizar treinamento adicional para obter métricas na última ronda
            start_time = time.time()
            train_loss = train(self.model, self.train_loader, epochs=self.epochs, criterion=self.criterion, device=self.device)
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
        val_loss, val_rmse, _, _, val_accuracy, val_precision, val_recall, val_f1 = evaluate(self.model, self.val_loader, self.criterion, self.device)
        
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
        test_loss, test_rmse, _, _, test_accuracy, test_precision, test_recall, test_f1 = evaluate(self.model, self.test_loader, self.criterion, self.device)
        
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
                background_data_np = None
                feature_names_list = []
                X_background_path = None

                if hasattr(self, 'X_train_df_scaled') and self.X_train_df_scaled is not None and not self.X_train_df_scaled.empty:
                    background_data_np = self.X_train_df_scaled.to_numpy()
                    MAX_BACKGROUND_SAMPLES = 500
                    if background_data_np.shape[0] > MAX_BACKGROUND_SAMPLES:
                        logger.info(f"Amostrando dados de fundo de {background_data_np.shape[0]} para {MAX_BACKGROUND_SAMPLES} instâncias.")
                        indices = np.random.choice(background_data_np.shape[0], MAX_BACKGROUND_SAMPLES, replace=False)
                        background_data_np = background_data_np[indices]
                    
                    X_background_path = self.base_reports / "X_train_background.npy"
                    np.save(X_background_path, background_data_np)
                    logger.info(f"Dados de fundo (self.X_train_df_scaled) para explicabilidade salvos em {X_background_path} com {background_data_np.shape[0]} instâncias.")
                    feature_names_list = list(self.X_train_df_scaled.columns)
                else:
                    logger.warning("self.X_train_df_scaled não está disponível/vazio. Usando fallback: 1º batch do train_loader para dados de fundo.")
                    if self.train_loader and len(self.train_loader.dataset) > 0:
                        first_train_batch_features, _ = next(iter(self.train_loader))
                        background_data_np = first_train_batch_features.numpy()
                        X_background_path = self.base_reports / "X_train_batch_fallback_background.npy"
                        np.save(X_background_path, background_data_np)
                        logger.info(f"Dados de fundo (fallback: X_train batch) para explicabilidade salvos em {X_background_path}")
                        num_features = background_data_np.shape[1] if len(background_data_np.shape) > 1 else 0
                        feature_names_list = [f'feature_{i}' for i in range(num_features)]
                    else:
                        logger.error("Train_loader vazio ou não disponível. Não é possível obter dados de fundo para explicabilidade.")
                        # Define background_data_np como um array vazio para evitar erros subsequentes se não houver dados
                        background_data_np = np.array([]) 
                        X_background_path = self.base_reports / "empty_background.npy"
                        np.save(X_background_path, background_data_np) # Salva um array vazio

                if not feature_names_list and background_data_np.size > 0: # Se nomes não definidos mas dados existem
                     num_features = background_data_np.shape[1] if len(background_data_np.shape) > 1 else 0
                     feature_names_list = [f'feature_generic_{i}' for i in range(num_features)]
                elif not feature_names_list and background_data_np.size == 0:
                    logger.warning("Não foi possível determinar nomes de features pois não há dados de fundo.")
                    feature_names_list = ['fallback_feature'] # Evitar erro no ModelExplainer

                explainer = ModelExplainer(self.model, feature_names=feature_names_list, device=self.device)
                if X_background_path and background_data_np.size > 0:
                    lime_duration, shap_duration = generate_explainability(explainer, X_background_path, self.base_reports)
                else:
                    logger.warning("Skipping LIME/SHAP generation due to missing background data or path.")
                
                # IMPLEMENTAÇÃO ROBUSTA: Garantir que o arquivo explained_instance_info.json seja criado
                try:
                    instance_to_explain_normalized = None
                    original_target_value_for_explained_instance = None
                    
                    # Selecionar aleatoriamente uma instância do conjunto de teste para explicar
                    random_idx = -1 # Inicializar para o caso de não se encontrar nenhuma instância
                    if self.test_loader and self.test_loader.dataset and len(self.test_loader.dataset) > 0:
                        num_test_samples = len(self.test_loader.dataset)
                        random_idx = random.randint(0, num_test_samples - 1)
                        
                        # CustomDataset[idx] retorna (feature_tensor, target_tensor)
                        instance_tensor_for_model, original_target_tensor = self.test_loader.dataset[random_idx]
                        
                        instance_to_explain_normalized = None
                        original_target_value_for_explained_instance = None

                        if instance_tensor_for_model is not None and original_target_tensor is not None:
                            try:
                                instance_to_explain_normalized = instance_tensor_for_model.cpu().numpy()
                                original_target_value_for_explained_instance = original_target_tensor.item()
                                logger.info(f"Instância aleatória {random_idx} (de {num_test_samples} amostras) selecionada e convertida do conjunto de teste para explicabilidade.")
                            except AttributeError as ae:
                                logger.error(f"Erro de atributo ao converter instância/alvo para explicabilidade (AttributeError): {ae}. "
                                             f"Verifique se instance_tensor_for_model (tipo: {type(instance_tensor_for_model)}) e "
                                             f"original_target_tensor (tipo: {type(original_target_tensor)}) são tensores válidos.", exc_info=True)
                            except Exception as e_conv:
                                logger.error(f"Erro inesperado ao converter instância/alvo para explicabilidade: {e_conv}", exc_info=True)
                        else:
                            logger.warning(f"instance_tensor_for_model ou original_target_tensor é None ANTES da tentativa de conversão. "
                                           f"instance_tensor_for_model tipo: {type(instance_tensor_for_model)}, "
                                           f"original_target_tensor tipo: {type(original_target_tensor)}.")

                        # Verificar se a instância e o seu alvo original foram obtidos e convertidos com sucesso
                        if instance_to_explain_normalized is not None and original_target_value_for_explained_instance is not None:
                            # Processar e guardar as informações da instância
                            current_instance_tensor_for_processing = torch.tensor(instance_to_explain_normalized, dtype=torch.float32).to(self.device)
                            with torch.no_grad():
                                logits = self.model(current_instance_tensor_for_processing.unsqueeze(0))
                                probability = torch.sigmoid(logits).item()
                            
                            predicted_class_value = 1 if probability >= 0.5 else 0
                            is_correct = (predicted_class_value == int(original_target_value_for_explained_instance))

                            instance_info = {
                                "source_dataset": "test_set_random_instance",
                                "random_instance_index_in_test_set": random_idx,
                                "prediction": float(round(probability, 5)),
                                "predicted_class": "Ataque" if probability >= 0.5 else "Normal",
                                "original_target_class": "Ataque" if original_target_value_for_explained_instance == 1 else "Normal",
                                "prediction_correct": bool(is_correct),
                                "prediction_confidence": float(round(max(probability, 1 - probability) * 100, 2)),
                                "feature_count": len(instance_to_explain_normalized),
                                "features_normalized": instance_to_explain_normalized.tolist(),
                            }
                            
                            instance_info_path = self.base_reports / "explained_instance_info.json"
                            with open(instance_info_path, 'w') as f:
                                json.dump(instance_info, f, indent=4, default=str)
                            logger.info(f"Informações da instância de explicabilidade (do test_loader) salvas em {instance_info_path}")
                        else:
                            # Log se instance_to_explain_normalized ou original_target_value_for_explained_instance for None
                            logger.warning("Falha ao obter/converter a instância de teste normalizada e/ou o seu valor alvo original. 'explained_instance_info.json' pode não ser gerado ou estar incompleto.")
                    else:
                        logger.warning("Test_loader vazio ou não disponível. Não é possível gerar explained_instance_info.json.")

                except Exception as e:
                    logger.error(f"Erro ao gerar explained_instance_info.json OU ao criar o arquivo: {e}", exc_info=True)
                
                # Gerar versão formatada do arquivo LIME para melhor leitura
                # NOVO Bloco INICIO
                try:
                    # 'instance_to_explain_normalized' e 'instance_info' devem estar definidos
                    # nas secções anteriores do método 'evaluate'.
                    # 'background_data_np' também deve estar definido (dados de treino/fundo).
                    # 'feature_names_list' é usado para criar o current_explainer.

                    if instance_to_explain_normalized is not None and background_data_np is not None:
                        current_explainer = ModelExplainer(self.model, feature_names=feature_names_list, device=self.device)
                        
                        prepared_original_target_info = {
                            'original_target_class': instance_info.get('original_target_class'),
                            'original_target': 1 if instance_info.get('original_target_class') == "Ataque" else 0,
                            'prediction_correct': instance_info.get('prediction_correct'),
                            'predicted_raw_value': instance_info.get('prediction')
                        }

                        # Nota: A função generate_explainability_formatado retorna (lime_duration, shap_duration)
                        # A variável lime_duration já existe neste escopo devido à chamada anterior a generate_explainability.
                        # Se precisar da duração específica desta chamada formatada, use um nome de variável diferente.
                        _, _ = generate_explainability_formatado(
                            explainer=current_explainer,
                            X_background_path_or_data=background_data_np,
                            base_reports=self.base_reports,
                            instance_to_explain_node=instance_to_explain_normalized,
                            scaler=self.scaler, # Assegure-se que self.scaler está disponível e é o correto
                            feature_names=feature_names_list,
                            original_target_info=prepared_original_target_info
                        )
                        logger.info(f"Explicação LIME formatada gerada com sucesso para cliente {self.client_id}")
                    else:
                        if instance_to_explain_normalized is None:
                            logger.warning("instance_to_explain_normalized é None. Não é possível gerar LIME formatado.")
                        if background_data_np is None:
                            logger.warning("background_data_np é None. Não é possível gerar LIME formatado.")

                except Exception as e:
                    logger.error(f"Erro ao gerar explicação LIME formatada: {e}", exc_info=True)
                # NOVO Bloco FIM                
                # Adicionar durações à história
                self.history["lime_duration_seconds"].append(lime_duration)
                self.history["shap_duration_seconds"].append(shap_duration)
            except Exception as e:
                logger.error(f"Error during explainability generation: {e}", exc_info=True)
        
        logger.info(f"CLIENT {self.cid}: EVALUATE - Entering report generation section for round {server_round}.")
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
                try:
                    logger.info(f"Iniciando geração do relatório HTML para cliente {args.cid}")
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
                    logger.info(f"Relatório HTML gerado com sucesso para cliente {args.cid}")
                except Exception as e:
                    logger.error(f"Erro específico ao gerar relatório HTML: {e}", exc_info=True)
                
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
        logger.info(f"CLIENT {self.cid}: CLFEClient.evaluate - END")
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
    logger.info(f"CLIENT {args.cid}: Main execution block - START. Args: cid={args.cid}, server={args.server_address}, epochs={args.epochs}, seed={args.seed}, dataset_path={args.dataset_path}")
    # Criando a instância do cliente
    client = CLFEClient(model, train_loader, val_loader, test_loader, args.epochs, device, pos_weight_value,
                        X_train_df_scaled, feature_names, scaler) # Passar novos argumentos
    
    # Iniciar a conexão com o servidor Flower
    logger.info(f"CLIENT {args.cid}: Initializing Flower client connection to server at {args.server_address}")
    fl.client.start_numpy_client(server_address=args.server_address, client=client)
    
    logger.info("Federated learning finished.")
