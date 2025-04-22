import os
import warnings
import flwr as fl
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import export_text
import matplotlib.pyplot as plt
import json
import utils as u

# Ignorar avisos
warnings.filterwarnings('ignore')

# Configurações
DEVICE = u.DEVICE
INPUT_SIZE = 9  # Número de features no dataset
VALIDATION_SPLIT = 0.1
BATCH_SIZE = 32

def load_data():
    """Carrega e prepara os dados para treinamento."""
    # Carregar o dataset
    df = pd.read_csv('dsCaliforniaHousing/housing.csv')
    
    # Separar features e target
    X = df.drop('median_house_value', axis=1)
    y = df['median_house_value']
    
    # Codificar a variável categórica 'ocean_proximity'
    le = LabelEncoder()
    X['ocean_proximity'] = le.fit_transform(X['ocean_proximity'])
    
    # Normalizar os dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Dividir dados de treino em treino e validação
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=VALIDATION_SPLIT, random_state=42
    )
    
    # Converter para tensores do PyTorch
    train_features = torch.FloatTensor(X_train)
    train_targets = torch.FloatTensor(y_train.values)
    val_features = torch.FloatTensor(X_val)
    val_targets = torch.FloatTensor(y_val.values)
    test_features = torch.FloatTensor(X_test)
    test_targets = torch.FloatTensor(y_test.values)
    
    # Criar DataLoaders
    train_loader = DataLoader(
        TensorDataset(train_features, train_targets),
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(val_features, val_targets),
        batch_size=BATCH_SIZE
    )
    test_loader = DataLoader(
        TensorDataset(test_features, test_targets),
        batch_size=BATCH_SIZE
    )
    
    return train_loader, val_loader, test_loader

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, input_size, cid, trainloader, valloader, testloader, num_rounds=5):
        self.cid = cid
        self.input_size = input_size
        self.model = u.DecisionTreeModel(input_size)
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

    def get_parameters(self, config=None):
        return u.get_parameters(self.model)

    def set_parameters(self, parameters):
        u.set_parameters(self.model, parameters)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        train_loss = u.train(self.model, self.trainloader)
        validation_loss, validation_rmse = u.test(self.model, self.valloader)
        
        # Armazenar métricas
        self.current_metrics['training_loss'] = train_loss
        self.current_metrics['validation_loss'] = validation_loss
        self.current_metrics['rmse'] = validation_rmse
        
        # Gerar e salvar visualização da árvore
        self._save_tree_visualization()
        
        print(f"[Client {self.cid}] Validation Loss: {validation_loss}")
        print(f"[Client {self.cid}] Validation RMSE: {validation_rmse}")
        return self.get_parameters(), len(self.trainloader.dataset), {"loss": float(validation_loss)}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        loss, rmse = u.test(self.model, self.testloader)
        
        # Armazenar métricas de avaliação
        self.current_metrics['evaluation_loss'] = loss
        self.current_metrics['rmse'] = rmse
        
        print(f"[Client {self.cid}] Evaluation Loss: {loss}")
        print(f"[Client {self.cid}] RMSE: {rmse}")
        
        # Incrementar o contador de rounds ANTES de gerar explicações
        self.current_round += 1
        
        # Salvar o modelo e gerar explicações
        self._save_model()
        self.generate_explanations()
        
        return float(loss), len(self.testloader.dataset), {"loss": float(loss), "rmse": float(rmse)}

    def _save_model(self):
        """Saves the trained model"""
        model_path = os.path.join('results', f'model_client_{self.cid}.pkl')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        import pickle
        with open(model_path, 'wb') as f:
            pickle.dump(self.model.tree, f)

    def _save_tree_visualization(self):
        """Saves tree visualization and feature importance"""
        # Save tree text
        tree_text = export_text(self.model.tree, feature_names=self.feature_names)
        with open(f'reports/tree_structure_round_{self.current_round}.txt', 'w') as f:
            f.write(tree_text)
        
        # Save feature importance plot
        plt.figure(figsize=(10, 6))
        importances = self.model.feature_importances_
        if importances is not None:
            plt.bar(self.feature_names, importances)
            plt.xticks(rotation=45, ha='right')
            plt.title('Feature Importance')
            plt.tight_layout()
            plt.savefig(f'reports/feature_importance_round_{self.current_round}.png')
            plt.close()

    def generate_explanations(self):
        """Generates explanations using tree visualizations"""
        print(f"[Client {self.cid}] Generating explanations...")
        
        try:
            # Save current round data
            print(f"[Client {self.cid}] Saving data for round {self.current_round}...")
            try:
                self._save_round_data()
                
                # Generate final report after last round
                if self.current_round == self.num_rounds:
                    print(f"[Client {self.cid}] Generating final report...")
                    final_report_path = self.generate_final_report()
                    print(f"[Client {self.cid}] Final report generated: {final_report_path}")
                    
            except Exception as e:
                print(f"[Client {self.cid}] Error saving data/generating report: {str(e)}")
                import traceback
                print(traceback.format_exc())
                raise e
            
        except Exception as e:
            print(f"[Client {self.cid}] Error during explanation generation: {str(e)}")
            raise e

    def _save_round_data(self):
        """Salva os dados da round atual no arquivo JSON"""
        timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
        
        round_data = {
            'round': self.current_round,
            'timestamp': timestamp,
            'metrics': {
                'training_loss': float(self.current_metrics.get('training_loss', 0)),
                'validation_loss': float(self.current_metrics.get('validation_loss', 0)),
                'evaluation_loss': float(self.current_metrics.get('evaluation_loss', 0)),
                'rmse': float(self.current_metrics.get('rmse', 0))
            },
            'visualizations': {
                'tree_structure': f'tree_structure_round_{self.current_round}.txt',
                'feature_importance': f'feature_importance_round_{self.current_round}.png'
            }
        }
        
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        # Atualizar ou adicionar dados da round
        data['rounds'] = [r for r in data['rounds'] if r['round'] != self.current_round]
        data['rounds'].append(round_data)
        data['rounds'].sort(key=lambda x: x['round'])
        
        with open(self.json_path, 'w') as f:
            json.dump(data, f, indent=4)

    def generate_final_report(self):
        """Gera o relatório final em HTML com visualizações da árvore"""
        # Gerar visualização da árvore para a última round
        self._save_tree_visualization()
        
        with open(self.json_path, 'r') as f:
            data = json.load(f)
        
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Final Report - Decision Tree</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3 {{ color: #333; }}
                .metrics {{ margin: 20px 0; }}
                .round {{ margin: 30px 0; padding: 20px; border: 1px solid #ddd; }}
                .timestamp {{ color: #666; font-style: italic; }}
                pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
                img {{ max-width: 100%; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Final Report - Client {self.cid}</h1>
        """
        
        # Add sections for each round
        for round_data in data['rounds']:
            html_template += f"""
            <div class="round">
                <h2>Round {round_data['round']}</h2>
                <div class="timestamp">Timestamp: {round_data['timestamp']}</div>
                
                <div class="metrics">
                    <h3>Metrics</h3>
                    <ul>
                        <li>Training Loss: {round_data['metrics']['training_loss']:.2f}</li>
                        <li>Validation Loss: {round_data['metrics']['validation_loss']:.2f}</li>
                        <li>Evaluation Loss: {round_data['metrics']['evaluation_loss']:.2f}</li>
                        <li>RMSE: {round_data['metrics']['rmse']:.2f}</li>
                    </ul>
                </div>
                
                <h3>Tree Structure</h3>
                <pre>{open(f"reports/{round_data['visualizations']['tree_structure']}").read()}</pre>
                
                <h3>Feature Importance</h3>
                <img src="{round_data['visualizations']['feature_importance']}" 
                     alt="Feature Importance - Round {round_data['round']}">
            </div>
            """
        
        html_template += """
        </body>
        </html>
        """
        
        report_path = 'reports/final_report.html'
        with open(report_path, 'w') as f:
            f.write(html_template)
        
        return report_path

def main():
    """Função principal."""
    # Carregar dados
    trainloader, valloader, testloader = load_data()
    
    # Inicializar cliente
    client = FlowerClient(
        input_size=INPUT_SIZE,
        cid='2',  # ID do cliente
        trainloader=trainloader,
        valloader=valloader,
        testloader=testloader
    )
    
    # Iniciar cliente Flower
    fl.client.start_numpy_client(
        server_address="server:9091",
        client=client,
    )

if __name__ == "__main__":
    main()
