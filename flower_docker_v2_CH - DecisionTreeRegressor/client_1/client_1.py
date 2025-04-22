import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, random_split
import flwr as fl
import utils as u
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import json

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

class CifarClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = LinearRegression()
        self.x_train, self.x_test, self.y_train, self.y_test = load_data()
        self.x_train_tensor = torch.FloatTensor(self.x_train)
        self.y_train_tensor = torch.FloatTensor(self.y_train)
        self.x_test_tensor = torch.FloatTensor(self.x_test)
        self.y_test_tensor = torch.FloatTensor(self.y_test)
        self.feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude', 'Distance']
        self._initialize_reports()
        
    def _initialize_reports(self):
        """Inicializa o arquivo rounds_data.json no início do treinamento"""
        print("[Client 1] Inicializando arquivo de dados das rounds...")
        os.makedirs('reports', exist_ok=True)
        json_path = os.path.join('reports', 'rounds_data.json')
        
        # Criar estrutura inicial do arquivo
        initial_data = {
            'rounds': []
        }
        
        # Salvar arquivo inicial
        with open(json_path, 'w') as f:
            json.dump(initial_data, f, indent=4)
        print("[Client 1] Arquivo rounds_data.json inicializado com sucesso.")

    def train(self, train_loader, val_loader, num_rounds=5):
        """
        Treina o modelo com os dados fornecidos.
        
        Parâmetros:
        - train_loader, val_loader: carregadores de dados para treino e validação
        - num_rounds: número de rondas de treino
        
        Retorna:
        - Métricas finais após o treino
        """
        # Inicializar listas para armazenar métricas
        losses_train = []
        losses_val = []
        
        # Para cada ronda de treino
        for i in range(1, num_rounds + 1):
            # Treinar o modelo com os dados de treino
            loss_train = self.client.fit_round(i)
            losses_train.append(loss_train)
            
            # Validar o modelo com os dados de validação
            loss_val, _ = self.client.validate()
            losses_val.append(loss_val)
            
            # Salvar os dados da ronda atual
            self.client.save_round_data(i, loss_train, loss_val)
        
        # Avaliar o modelo final
        test_metrics = self.client.evaluate()
        
        # Retornar os valores finais após todo o treino e validação
        final_train_loss = losses_train[-1]
        final_val_loss = losses_val[-1]
        final_eval_loss, final_rmse = test_metrics
        
        # Gerar relatório final (se estiver no final do processo)
        if not self.done_training:
            print("Gerando relatório final...")
            self.client.generate_final_report()
            self.done_training = True
        
        # Retornar as métricas finais
        return final_train_loss, final_val_loss, final_eval_loss, final_rmse

    def get_parameters(self):
        return self.model.coef_, self.model.intercept_

    def fit(self, parameters, config):
        self.model.coef_, self.model.intercept_ = parameters
        losses_train, losses_val, test_metrics = self.train(self.x_train_tensor, self.y_train_tensor, self.x_test_tensor, self.y_test_tensor)
        return self.get_parameters(), len(self.x_train_tensor), {}

    def evaluate(self, parameters, config):
        self.model.coef_, self.model.intercept_ = parameters
        losses_train, losses_val, test_metrics = self.train(self.x_train_tensor, self.y_train_tensor, self.x_test_tensor, self.y_test_tensor)
        # Retornar os valores finais após todo o treino e validação
        final_train_loss = losses_train[-1]
        final_val_loss = losses_val[-1]
        final_eval_loss, final_rmse = test_metrics
        
        # Gerar relatório final
        # Nota: O relatório já é gerado no método train()
        
        # Retornar as métricas finais
        return final_train_loss, len(self.x_test_tensor), {"mse": final_eval_loss}

# Criar o cliente Flower
client = u.FlowerClient(
    input_size=input_size,
    cid=1,
    trainloader=trainloaders[0],
    valloader=valloaders[0],
    testloader=testloader,
)

# Atualizar nomes das features
client.feature_names = feature_names

# Iniciar o cliente
fl.client.start_numpy_client(server_address="server:9091", client=client)
