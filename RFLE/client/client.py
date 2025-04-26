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

# Configurar logging detalhado
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][Client %(cid)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# Argumentos do script
parser = argparse.ArgumentParser(description="RLFE Federated Client")
parser.add_argument('--cid', type=int, required=True, help='ID do cliente (1-indexed)')
parser.add_argument('--num_clients', type=int, required=True, help='Número de clientes a executar neste teste')
parser.add_argument('--num_total_clients', type=int, default=10, help='Número total de clientes para particionamento dos dados')
parser.add_argument('--dataset_path', type=str, default="../DatasetIOT/transformed_dataset_imeisv_8642840401612300.csv", help="Path to dataset CSV")
parser.add_argument('--seed', type=int, default=42, help='Seed para reprodutibilidade')
parser.add_argument('--epochs', type=int, default=5, help="Epochs per round")
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
logging.info(f"Loading dataset from {args.dataset_path}", extra={'cid': args.cid})
df = pd.read_csv(args.dataset_path)

# --- Validação e limpeza de dados ---
# 1. Remover colunas constantes (zero variância)
const_cols = [col for col in df.columns if df[col].nunique() == 1]
if const_cols:
    logging.info(f"Removing constant columns: {const_cols}", extra={'cid': args.cid})
    df = df.drop(columns=const_cols)

# 2. Remover colunas de identificação e tempo (não preditivas)
irrelevant_cols = ['índice', '_time', 'imeisv']
cols_to_remove = [col for col in irrelevant_cols if col in df.columns]
if cols_to_remove:
    logging.info(f"Removing identifier/time columns: {cols_to_remove}", extra={'cid': args.cid})
    df = df.drop(columns=cols_to_remove)

# 2.1. Extração de features temporais a partir de _time (se existir)
if '_time' in df.columns:
    logging.info("Extracting temporal features from _time", extra={'cid': args.cid})
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
    logging.info(f"Filling missing values in columns: {missing_cols}", extra={'cid': args.cid})
    for col in missing_cols:
        mediana = df[col].median()
        df[col] = df[col].fillna(mediana)

# 4. Codificação de variáveis categóricas (one-hot encoding)
categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
if categorical_cols:
    logging.info(f"Applying one-hot encoding to columns: {categorical_cols}", extra={'cid': args.cid})
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
# --- Fim da validação e limpeza ---

# Particionamento on-the-fly
# O dataset é sempre dividido por num_total_clients, independentemente do número de clientes executados
partes = np.array_split(df, args.num_total_clients)
client_data = partes[args.cid - 1]  # Numeração começa em 1
logging.info(f"Client data shape: {client_data.shape}", extra={'cid': args.cid})

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

# Treino
logging.info("Starting training...", extra={'cid': args.cid})
train_loss = train(model, train_loader, epochs=args.epochs, device=device)
val_loss, val_rmse, _, _ = evaluate(model, val_loader, device=device)
test_loss, test_rmse, y_pred, y_true = evaluate(model, test_loader, device=device)
logging.info(f"Train loss: {train_loss:.4f}", extra={'cid': args.cid})
logging.info(f"Val loss: {val_loss:.4f} | Val RMSE: {val_rmse:.2f}", extra={'cid': args.cid})
logging.info(f"Test loss: {test_loss:.4f} | Test RMSE: {test_rmse:.2f}", extra={'cid': args.cid})

# Explicabilidade
logging.info("Generating LIME explanation...", extra={'cid': args.cid})
lime_exp = explainer.explain_lime(X_train, X_test[0])
lime_path = base_reports / "lime_explanation.txt"
with open(lime_path, "w") as f:
    f.write(str(lime_exp.as_list()))
logging.info(f"LIME explanation saved to {lime_path}", extra={'cid': args.cid})

logging.info("Generating SHAP explanation...", extra={'cid': args.cid})
shap_values = explainer.explain_shap(X_train, n_samples=50)
shap_path = base_reports / "shap_values.npy"
np.save(shap_path, shap_values)
logging.info(f"SHAP values saved to {shap_path}", extra={'cid': args.cid})

# Guardar info.txt e métricas
with open(base_reports / "info.txt", "w") as f:
    f.write(f"Client {args.cid} - Num samples: {len(client_data)}\n")
    f.write(f"Train loss: {train_loss:.4f}\nVal loss: {val_loss:.4f}\nVal RMSE: {val_rmse:.2f}\nTest loss: {test_loss:.4f}\nTest RMSE: {test_rmse:.2f}\n")

logging.info("Pipeline finished.", extra={'cid': args.cid})
