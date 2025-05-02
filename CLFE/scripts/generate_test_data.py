#!/usr/bin/env python
"""
Script para gerar dados de teste para explicabilidade com o formato correto (965 features).
Este arquivo cria arrays numpy compatíveis com o modelo treinado para testes.
"""
import os
import sys
import numpy as np
import pandas as pd

# Adicionar diretório pai ao PYTHONPATH para importações relativas
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, parent_dir)

# Configurar caminhos
output_dir = "client/reports/client_1"
os.makedirs(output_dir, exist_ok=True)

# Número de features esperado pelo modelo
num_features = 965

# Tamanho do conjunto de teste
num_samples = 100

# Gerar dados de treinamento aleatórios
print(f"Gerando conjunto de dados com {num_samples} amostras e {num_features} features...")
X_train = np.random.randn(num_samples, num_features).astype(np.float32)
y_train = np.random.randint(0, 2, num_samples).astype(np.float32)

# Salvar dados
X_train_path = os.path.join(output_dir, "X_train.npy")
y_train_path = os.path.join(output_dir, "y_train.npy")

print(f"Salvando X_train em {X_train_path}...")
np.save(X_train_path, X_train)

print(f"Salvando y_train em {y_train_path}...")
np.save(y_train_path, y_train)

# Gerar um arquivo de texto de exemplo para a extração de nomes de features
fake_explanation_text = """
## TOP FEATURES 
1. dl_bitrate_0 = 0.4567
2. dayofweek = 3
3. dl_bitrate_1 = 1.234
4. dl_bitrate_2 = 0.987
5. is_weekend = 0
6. hour = 14
7. month = 5
8. cell_x_dl_tx_10 = 0.456
9. ul_bitrate_5 = 0.789
10. cell_x_ul_retx_7 = 0.123
"""

explanation_path = os.path.join(output_dir, "lime_explanation.txt")
print(f"Salvando explicação de exemplo em {explanation_path}...")
with open(explanation_path, "w") as f:
    f.write(fake_explanation_text)

print("Concluído!")
