from collections import OrderedDict
from typing import List
import torch
import torch.nn as nn
import numpy as np
import flwr as fl
import pandas as pd
from explainability import ModelExplainer
import os
import json

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Número de clientes para aprendizado distribuído
NUM_CLIENTS = 2

# Criação do modelo
class LinearRegressionModel(nn.Module):
    def __init__(self, input_size: int, output_size: int = 1) -> None:
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

# Definição da classe FlowerClient
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, input_size, cid, trainloader, valloader, testloader, num_rounds=5):
        self.cid = cid
        self.input_size = input_size
        self.model = LinearRegressionModel(input_size).to(DEVICE)
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.current_metrics = {}
        self.current_round = 0
        self.num_rounds = num_rounds
        
        # Criar diretórios para relatórios se não existirem
        os.makedirs('reports', exist_ok=True)
        os.makedirs('results', exist_ok=True)
        
        # Inicializar arquivo JSON para armazenar dados das rounds
        self.json_path = os.path.join('reports', 'rounds_data.json')
        if not os.path.exists(self.json_path):
            with open(self.json_path, 'w') as f:
                json.dump({'rounds': []}, f)

        # Definir nomes das features para explicabilidade
        self.feature_names = [
            'longitude', 'latitude', 'housing_median_age', 'total_rooms',
            'total_bedrooms', 'population', 'households', 'median_income',
            'ocean_proximity'
        ]
        
        # Inicializar o explainer uma única vez
        self.explainer = ModelExplainer(
            model=self.model,
            feature_names=self.feature_names,
            cid=self.cid
        )
        
        # Definir dados de teste para o explainer
        self.explainer.set_test_data(self.testloader)

    def get_parameters(self, config=None):
        return get_parameters(self.model)

    def set_parameters(self, parameters):
        set_parameters(self.model, parameters)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        train_loss = train(self.model, self.trainloader, epochs=1)
        validation_loss, validation_rmse = test(self.model, self.valloader)
        
        # Armazenar métricas
        self.current_metrics['training_loss'] = train_loss
        self.current_metrics['validation_loss'] = validation_loss
        self.current_metrics['rmse'] = validation_rmse
        
        print(f"[Client {self.cid}] Validation Loss: {validation_loss}")
        print(f"[Client {self.cid}] Validation RMSE: {validation_rmse}")
        return self.get_parameters(), len(self.trainloader.dataset), {"loss": float(validation_loss)}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        loss, rmse = test(self.model, self.testloader)
        
        # Armazenar métricas de avaliação
        self.current_metrics['evaluation_loss'] = loss
        self.current_metrics['rmse'] = rmse
        
        print(f"[Client {self.cid}] Evaluation Loss: {loss}")
        print(f"[Client {self.cid}] RMSE: {rmse}")
        
        # Salvar o modelo após cada avaliação
        model_path = os.path.join('results', f'model_client_{self.cid}.pt')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_size': self.input_size,
            'loss': loss
        }, model_path)
        
        try:
            # Incrementar o contador de rounds
            self.current_round += 1
            
            # Gerar explicações após o treinamento final
            self.generate_explanations()
        except Exception as e:
            print(f"[Client {self.cid}] Erro ao gerar explicações: {str(e)}")
        
        return float(loss), len(self.testloader.dataset), {"loss": float(loss), "rmse": float(rmse)}

    def generate_explanations(self):
        """Gera explicações usando LIME e SHAP após o treinamento"""
        print(f"[Client {self.cid}] Gerando explicações...")
        
        # Usar o explainer já inicializado em vez de criar um novo
        # Passar métricas atuais para o explainer
        self.explainer.set_metrics(
            round_number=self.current_round,
            validation_loss=self.current_metrics.get('validation_loss'),
            evaluation_loss=self.current_metrics.get('evaluation_loss'),
            training_loss=self.current_metrics.get('training_loss'),
            rmse=self.current_metrics.get('rmse')
        )

        try:
            # Salvar dados da round atual
            print(f"[Client {self.cid}] Salvando dados da round {self.current_round}...")
            try:
                self.explainer.save_round_data(
                    round_number=self.current_round,
                    training_loss=self.current_metrics.get('training_loss'),
                    validation_loss=self.current_metrics.get('validation_loss'),
                    evaluation_loss=self.current_metrics.get('evaluation_loss'),
                    rmse=self.current_metrics.get('rmse')
                )
                
                # Gerar relatório final após a última round
                if self.current_round == self.num_rounds:
                    print(f"[Client {self.cid}] Gerando relatório final...")
                    final_report_path = self.explainer.generate_final_report()
                    print(f"[Client {self.cid}] Relatório final gerado: {final_report_path}")
                    
            except Exception as e:
                print(f"[Client {self.cid}] Erro ao salvar dados/gerar relatório: {str(e)}")
                import traceback
                print(traceback.format_exc())
                raise e
            
        except Exception as e:
            print(f"[Client {self.cid}] Erro durante a geração de explicações: {str(e)}")
            raise e

# Funções auxiliares
def get_parameters(model) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def set_parameters(model, parameters: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

def train(model, trainloader, epochs: int):
    """Treina o modelo de regressão linear."""
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    model.train()
    epoch_loss = 0.0
    for epoch in range(epochs):
        batch_losses = []
        for features, targets in trainloader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            predictions = model(features)
            loss = criterion(predictions, targets.view(-1, 1))
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        epoch_loss = sum(batch_losses) / len(batch_losses)
        print(f"[Client {model.cid if hasattr(model, 'cid') else '?'}] Epoch {epoch + 1}: Train Loss: {epoch_loss}")
    return epoch_loss

def test(model, testloader):
    """Avalia o modelo no conjunto de teste."""
    criterion = torch.nn.MSELoss()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    model.eval()
    with torch.no_grad():
        for features, targets in testloader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            predictions = model(features)
            total_loss += criterion(predictions, targets.view(-1, 1)).item()
            
            # Coletar predições e alvos para RMSE
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
            
    avg_loss = total_loss / len(testloader)
    
    # Calcular RMSE
    rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_targets)) ** 2))
    
    return avg_loss, rmse
