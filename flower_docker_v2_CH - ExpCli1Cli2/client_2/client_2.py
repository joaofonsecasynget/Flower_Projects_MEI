import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import flwr as fl
import utils as u
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import json
from explainability import ModelExplainer

class CustomDataset(Dataset):
    def __init__(self, features, targets):
        # Garantir que os dados estejam no formato correto
        if isinstance(features, pd.DataFrame):
            features = features.values
        if isinstance(targets, pd.Series):
            targets = targets.values
            
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def load_datasets(csv_path, num_clients):
    # Carregar e pré-processar dados
    df = pd.read_csv(csv_path)
    
    # Preencher valores ausentes
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].mean())
    
    # Converter ocean_proximity para valores numéricos
    mapping = {'<1H OCEAN': 1, 'INLAND': 2, 'NEAR OCEAN': 3, 'NEAR BAY': 4, 'ISLAND': 5}
    df['ocean_proximity'] = df['ocean_proximity'].map(mapping)

    # Separar features e target
    features = df.drop(['median_house_value'], axis=1)
    target = df['median_house_value']

    # Normalizar features
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    y = target

    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Criar dataset de treino
    train_dataset = CustomDataset(X_train, y_train)

    # Dividir dados de treino entre os clientes
    partition_size = len(train_dataset) // num_clients
    lengths = [partition_size] * num_clients
    train_subsets = random_split(
        train_dataset,
        lengths,
        generator=torch.Generator().manual_seed(42)
    )

    # Criar dataloaders para treino e validação
    trainloaders = []
    valloaders = []

    for subset in train_subsets:
        # Dividir cada subset em treino e validação
        len_val = len(subset) // 10  # 10% para validação
        len_train = len(subset) - len_val
        train_subset, val_subset = random_split(
            subset,
            [len_train, len_val],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Criar dataloaders
        trainloaders.append(DataLoader(train_subset, batch_size=32, shuffle=True))
        valloaders.append(DataLoader(val_subset, batch_size=32, shuffle=False))

    # Criar dataloader para teste
    test_dataset = CustomDataset(X_test, y_test)
    testloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Retornar também os nomes das features
    feature_names = features.columns.tolist()
    return trainloaders, valloaders, testloader, feature_names

# Carregar dados e criar dataloaders
csv_path = "./dsCaliforniaHousing/housing.csv"
trainloaders, valloaders, testloader, feature_names = load_datasets(csv_path, u.NUM_CLIENTS)

# Obter dimensão de entrada do modelo
dataset_base = trainloaders[0].dataset.dataset.dataset
input_size = dataset_base.features.shape[1]

# Criar o cliente Flower
client = u.FlowerClient(
    input_size=input_size,
    cid=2,
    trainloader=trainloaders[1],
    valloader=valloaders[1],
    testloader=testloader,
)

# Atualizar nomes das features
client.feature_names = feature_names

# Iniciar o cliente
fl.client.start_numpy_client(server_address="server:9091", client=client)
