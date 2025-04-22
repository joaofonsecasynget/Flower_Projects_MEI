import lime
import lime.lime_tabular
import shap
import numpy as np
import matplotlib.pyplot as plt
import torch
from jinja2 import Template
import os
import json
from datetime import datetime
import locale
import pandas as pd

# Configurar locale para português
try:
    locale.setlocale(locale.LC_TIME, 'pt_BR.UTF-8')
except:
    try:
        locale.setlocale(locale.LC_TIME, 'pt_PT.UTF-8')
    except:
        pass  # Manter o locale padrão se não encontrar português

class ModelExplainer:
    def __init__(self, model, feature_names, class_names=None, cid=None):
        self.model = model
        self.feature_names = feature_names
        self.class_names = class_names
        self.metrics = {}
        self.X_test = None  # Dados de teste
        self.X_test_data = None  # Dados de teste em formato numpy
        self.cid = cid  # ID do cliente
        self.lime_instance = None  # Instância fixa para LIME
        self.lime_instance_idx = None  # Índice da instância fixa
        self.lime_explainer = None  # Explainer LIME persistente
        self.initialized = False  # Flag para controlar a inicialização

    def set_metrics(self, round_number, validation_loss, evaluation_loss, training_loss=None, rmse=None):
        """Armazena as métricas para a round atual e inicializa o LIME explainer se necessário"""
        print(f"[DEBUG Client {self.cid}] set_metrics chamado para a round {round_number}, inicializado: {self.initialized}")
        
        if round_number not in self.metrics:
            self.metrics[round_number] = {}
        
        self.metrics[round_number].update({
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'evaluation_loss': evaluation_loss,
            'rmse': rmse
        })
        
        # Inicializar LIME explainer e selecionar instância fixa apenas na primeira round
        if round_number == 1 and not self.initialized and self.X_test is not None:
            print("Inicializando LIME explainer...")
            # Extrair dados do DataLoader se ainda não foram extraídos
            if self.X_test_data is None:
                for batch in self.X_test:
                    self.X_test_data = batch[0].numpy()
                    break
            
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                training_data=self.X_test_data,
                feature_names=self.feature_names,
                class_names=['price'],
                mode='regression'
            )
            # Selecionar e armazenar uma instância fixa para LIME
            self.lime_instance_idx = np.random.randint(0, self.X_test_data.shape[0])
            self.lime_instance = self.X_test_data[self.lime_instance_idx].copy()
            print(f"[Client {self.cid}] Selecionada instância fixa para LIME: índice {self.lime_instance_idx}")
            self.initialized = True
            print(f"[DEBUG Client {self.cid}] LIME explainer inicializado, initialized={self.initialized}")
    
    def set_test_data(self, X_test):
        """Define os dados de teste para geração de explicações"""
        self.X_test = X_test
        # Extrair dados do DataLoader
        for batch in X_test:
            self.X_test_data = batch[0].numpy()
            break
        
    def predict_fn(self, x):
        """Função de predição para LIME e SHAP"""
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.model.linear.weight.device)
            predictions = self.model(x_tensor).cpu().numpy()
            return predictions.reshape(-1)  # Garantir que o output seja 1D
    
    def initialize_explainers(self, X):
        """Armazena X_test para uso posterior"""
        if self.X_test_data is None:
            self.X_test_data = X
        
    def explain_lime(self, X, num_features=5):
        """Gera explicações usando LIME com a instância fixa"""
        if not self.initialized:
            raise ValueError("LIME explainer não foi inicializado. Certifique-se de chamar set_metrics primeiro.")
        
        print(f"[Client {self.cid}] Gerando explicação LIME para a instância fixa (índice {self.lime_instance_idx})")
        exp = self.lime_explainer.explain_instance(
            data_row=self.lime_instance,
            predict_fn=self.predict_fn,
            num_features=num_features
        )
        
        return exp
    
    def explain_shap(self, X, n_samples=100):
        """Gera explicações usando SHAP"""
        # Selecionar subset para análise
        if X.shape[0] > n_samples:
            # Usar o mesmo random_state para consistência
            analysis_data = shap.sample(X, n_samples, random_state=42)
        else:
            analysis_data = X
            n_samples = X.shape[0]
            
        try:
            # Tentar usar KernelExplainer
            explainer = shap.KernelExplainer(
                self.predict_fn,
                analysis_data,
                feature_names=self.feature_names
            )
            
            # Calcular valores SHAP para os mesmos dados
            shap_values = explainer.shap_values(analysis_data)
            
            return shap_values, analysis_data
            
        except Exception as e:
            print(f"Erro ao gerar explicações SHAP: {str(e)}")
            return None
    
    def _save_visualizations(self, round_number):
        """Gera e salva as visualizações LIME e SHAP para a round atual."""
        print(f"[DEBUG Client {self.cid}] Estado do explainer - inicializado: {self.initialized}, lime_explainer: {self.lime_explainer is not None}, instance_idx: {self.lime_instance_idx}")
        
        if not self.initialized:
            print("Erro: LIME explainer não foi inicializado. Use set_metrics() primeiro.")
            return
        
        # Usar os dados já extraídos anteriormente
        if self.X_test_data is None and self.X_test is not None:
            # Se X_test_data ainda não foi extraído, mas X_test existe, extraímos agora
            for batch in self.X_test:
                self.X_test_data = batch[0].numpy()
                break
        
        if self.X_test_data is None:
            print("Erro: Dados de teste não estão disponíveis. Use set_test_data() antes de gerar visualizações.")
            return
        
        try:
            # Gera explicações LIME utilizando a instância fixa
            print("Gerando visualização LIME...")
            lime_exp = self.explain_lime(self.X_test_data)
            
            # Extrair e armazenar os valores de importância das features
            lime_feature_importances = {}
            for feature, importance in lime_exp.as_list():
                lime_feature_importances[feature] = importance
            
            # Armazenar as importâncias LIME para a round atual
            self.save_feature_importances(round_number, lime_feature_importances, explainer_type="lime")
            
            print(f"Gerando plot LIME com 5 features para a instância fixa (índice {self.lime_instance_idx})")
            plt.figure(figsize=(6, 4))
            lime_exp.as_pyplot_figure(label=1)
            plt.tight_layout()
            plt.subplots_adjust(left=0.3)
            plt.savefig(os.path.join('reports', f'lime_explanation_round{round_number}.png'), bbox_inches='tight', dpi=300)
            plt.close()
            
            # Gera explicações SHAP
            print("Gerando visualização SHAP...")
            shap_values, analysis_data = self.explain_shap(self.X_test_data)
            if shap_values is not None:
                # Extrair e armazenar os valores de importância SHAP
                shap_feature_importances = {}
                # Calcular a média absoluta dos valores SHAP para cada feature
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                for i, feature_name in enumerate(self.feature_names):
                    shap_feature_importances[feature_name] = float(mean_abs_shap[i])
                
                # Armazenar as importâncias SHAP para a round atual
                self.save_feature_importances(round_number, shap_feature_importances, explainer_type="shap")
                
                print("Gerando plot SHAP...")
                plt.figure(figsize=(6, 4))
                shap.summary_plot(shap_values, analysis_data, feature_names=self.feature_names, show=False)
                plt.tight_layout()
                plt.savefig(os.path.join('reports', f'shap_summary_round{round_number}.png'), bbox_inches='tight', dpi=300)
                plt.close()
            else:
                print("Erro: Não foi possível gerar explicações SHAP")
        except Exception as e:
            print(f"Erro ao gerar visualizações: {str(e)}")

    def save_feature_importances(self, round_number, feature_importances, explainer_type="lime"):
        """Salva as importâncias das features para uma determinada round.
        
        Args:
            round_number: Número da round
            feature_importances: Dicionário com as importâncias das features
            explainer_type: Tipo de explainer (lime ou shap)
        """
        # Carrega dados existentes ou cria novo arquivo
        json_path = os.path.join('reports', f'{explainer_type}_feature_importances.json')
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
            else:
                data = {}
        except json.JSONDecodeError:
            # Se houver erro ao ler o arquivo, cria um novo
            data = {}
        
        # Converte valores numpy para float nativo do Python se necessário
        for feature, importance in feature_importances.items():
            if hasattr(importance, 'item'):  # Se for um tipo numpy
                feature_importances[feature] = importance.item()
        
        # Atualiza os dados da round atual
        data[str(round_number)] = feature_importances
        
        # Salva os dados atualizados
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
            
        print(f"Importâncias das features ({explainer_type.upper()}) para a round {round_number} salvas em reports/{explainer_type}_feature_importances.json")

    def save_round_data(self, round_number, training_loss, validation_loss, evaluation_loss, rmse=None):
        """Salva os dados da round atual."""
        print(f"[Client {self.cid}] Salvando dados da round {round_number}...")
        print(f"RMSE recebido: {rmse}")  # Debug
        
        # Verifica e cria o diretório reports se necessário
        print("Verificando diretório reports...")
        os.makedirs('reports', exist_ok=True)
        print("Diretório reports verificado/criado com sucesso")
        
        # Define os caminhos dos arquivos
        print("Caminhos dos arquivos definidos:")
        lime_path = f"lime_explanation_round{round_number}.png"
        shap_path = f"shap_summary_round{round_number}.png"
        print(f"LIME: {lime_path}")
        print(f"SHAP: {shap_path}")
        
        # Gera as visualizações
        self._save_visualizations(round_number)
        
        # Adiciona timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Carrega dados existentes ou cria novo arquivo
        json_path = os.path.join('reports', 'rounds_data.json')
        try:
            if os.path.exists(json_path):
                with open(json_path, 'r') as f:
                    data = json.load(f)
            else:
                data = {'rounds': []}
        except json.JSONDecodeError:
            # Se houver erro ao ler o arquivo, cria um novo
            data = {'rounds': []}

        # Converte valores numpy para float nativo do Python
        training_loss = float(training_loss) if training_loss is not None else None
        validation_loss = float(validation_loss) if validation_loss is not None else None
        evaluation_loss = float(evaluation_loss) if evaluation_loss is not None else None
        rmse = float(rmse) if rmse is not None else None

        # Atualiza os dados da round atual
        round_data = {
            'round': round_number,
            'timestamp': timestamp,
            'metrics': {
                'training_loss': training_loss,
                'validation_loss': validation_loss,
                'evaluation_loss': evaluation_loss,
                'rmse': rmse
            },
            'visualizations': {
                'lime': lime_path,
                'shap': shap_path
            }
        }
        
        print(f"Dados da round {round_number}:")
        print(f"Training Loss: {training_loss}")
        print(f"Validation Loss: {validation_loss}")
        print(f"Evaluation Loss: {evaluation_loss}")
        print(f"RMSE: {rmse}")
        
        # Remove dados antigos da mesma round, se existirem
        data['rounds'] = [r for r in data['rounds'] if r['round'] != round_number]
        data['rounds'].append(round_data)
        
        # Ordena as rounds por número
        data['rounds'].sort(key=lambda x: x['round'])
        
        # Salva os dados atualizados
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)

    def generate_final_report(self):
        """Gera o relatório final em HTML."""
        print("Gerando relatório final...")
        
        # Carregar dados das rounds
        with open('reports/rounds_data.json', 'r') as f:
            data = json.load(f)
            
        # Filtrar rounds para excluir a round 0
        data['rounds'] = [round_data for round_data in data['rounds'] if round_data['round'] > 0]
        
        # Extrair valores das métricas para os gráficos
        rmse_values = [round_data['metrics']['rmse'] for round_data in data['rounds']]
        training_loss_values = [round_data['metrics']['training_loss'] for round_data in data['rounds']]
        validation_loss_values = [round_data['metrics']['validation_loss'] for round_data in data['rounds']]
        evaluation_loss_values = [round_data['metrics']['evaluation_loss'] for round_data in data['rounds']]
        rounds = range(1, len(rmse_values) + 1)

        print(f"Valores de RMSE encontrados: {rmse_values}")
        
        # Configuração dos subplots para métricas
        plt.figure(figsize=(12, 8))
        plt.subplot(2, 2, 1)
        plt.plot(rounds, training_loss_values, marker='o')
        plt.title('Evolução do Training Loss')
        plt.xlabel('Round')
        plt.ylabel('Training Loss')
        plt.grid(True)
        plt.xticks(rounds)

        plt.subplot(2, 2, 2)
        plt.plot(rounds, validation_loss_values, marker='o')
        plt.title('Evolução do Validation Loss')
        plt.xlabel('Round')
        plt.ylabel('Validation Loss')
        plt.grid(True)
        plt.xticks(rounds)

        plt.subplot(2, 2, 3)
        plt.plot(rounds, evaluation_loss_values, marker='o')
        plt.title('Evolução do Evaluation Loss')
        plt.xlabel('Round')
        plt.ylabel('Evaluation Loss')
        plt.grid(True)
        plt.xticks(rounds)

        plt.subplot(2, 2, 4)
        plt.plot(rounds, rmse_values, marker='o')
        plt.title('Evolução do RMSE')
        plt.xlabel('Round')
        plt.ylabel('RMSE')
        plt.grid(True)
        plt.xticks(rounds)

        plt.tight_layout()
        plt.savefig('reports/metrics_evolution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Gerar gráficos de evolução da explicabilidade
        self.generate_explainability_evolution_chart(explainer_type="lime")
        self.generate_explainability_evolution_chart(explainer_type="shap")
        
        # Template HTML
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Relatório Final de Treino - Cliente {self.cid}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1, h2, h3, h4 {{ color: #333; }}
                .metrics {{ margin-bottom: 30px; }}
                .round {{ margin-bottom: 40px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f5f5f5; }}
                .timestamp {{ color: #666; font-style: italic; margin-bottom: 10px; }}
                
                /* Ajustes específicos para as imagens */
                .metrics img {{ 
                    max-width: 800px; 
                    width: 100%; 
                    height: auto; 
                    margin: 20px 0;
                }}
                .explanation-img {{
                    max-width: 500px;
                    width: 100%;
                    height: auto;
                    margin: 10px 0;
                }}
                .explanations-container {{
                    display: flex;
                    justify-content: space-between;
                    flex-wrap: wrap;
                    gap: 20px;
                }}
                .explanation-section {{
                    flex: 1;
                    min-width: 300px;
                }}
            </style>
        </head>
        <body>
            <h1>Relatório Final de Treino - Cliente {self.cid}</h1>
            
            <div class="metrics">
                <h2>Evolução das Métricas</h2>
                <img src="metrics_evolution.png" alt="Metrics Evolution">
            </div>
            
            <div class="metrics">
                <h2>Evolução da Explicabilidade</h2>
                <p>Este gráfico mostra como a importância das 5 features mais relevantes evolui ao longo das rondas de treino.</p>
                <h3>LIME</h3>
                <img src="explainability_evolution_lime.png" alt="LIME Explainability Evolution">
                <h3>SHAP</h3>
                <img src="explainability_evolution_shap.png" alt="SHAP Explainability Evolution">
            </div>
            
            <h2>Detalhes por Round</h2>
        """
        
        # Adicionar seção para cada round
        for round_data in data['rounds']:
            round_number = round_data['round']
            metrics = round_data['metrics']
            timestamp = round_data.get('timestamp', 'Não registrado')
            
            html_template += f"""
            <div class="round">
                <h3>Round {round_number}</h3>
                <div class="timestamp">Data/Hora: {timestamp}</div>
                
                <table>
                    <tr>
                        <th>Métrica</th>
                        <th>Valor</th>
                    </tr>
                    <tr>
                        <td>Training Loss</td>
                        <td>{metrics['training_loss']}</td>
                    </tr>
                    <tr>
                        <td>Validation Loss</td>
                        <td>{metrics['validation_loss']}</td>
                    </tr>
                    <tr>
                        <td>Evaluation Loss</td>
                        <td>{metrics['evaluation_loss']}</td>
                    </tr>
                    <tr>
                        <td>RMSE</td>
                        <td>{metrics['rmse']}</td>
                    </tr>
                </table>
                
                <div class="explanations-container">
                    <div class="explanation-section">
                        <h4>Explicações LIME</h4>
                        <img src="{round_data['visualizations']['lime']}" alt="LIME Explanation Round {round_number}" class="explanation-img">
                    </div>
                    
                    <div class="explanation-section">
                        <h4>Explicações SHAP</h4>
                        <img src="{round_data['visualizations']['shap']}" alt="SHAP Summary Round {round_number}" class="explanation-img">
                    </div>
                </div>
            </div>
            """
        
        html_template += """
            </body>
        </html>
        """
        
        # Salvar o relatório HTML
        with open('reports/final_report.html', 'w') as f:
            f.write(html_template)
            
        return 'reports/final_report.html'
        
    def generate_explainability_evolution_chart(self, explainer_type="lime"):
        """Gera um gráfico mostrando a evolução da importância das features ao longo das rondas.
        
        Args:
            explainer_type: Tipo de explainer (lime ou shap)
        """
        # Carregar as importâncias das features
        json_path = os.path.join('reports', f'{explainer_type}_feature_importances.json')
        if not os.path.exists(json_path):
            print(f"Arquivo de importâncias de features ({explainer_type}) não encontrado. Não é possível gerar o gráfico de evolução.")
            return
        
        with open(json_path, 'r') as f:
            feature_importances = json.load(f)
        
        # Converter as chaves de string para int para ordenação
        round_numbers = sorted([int(key) for key in feature_importances.keys()])
        
        # Identificar as top 5 features da última ronda
        last_round = str(max(round_numbers))
        top_features = sorted(
            feature_importances[last_round].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:5]
        top_feature_names = [feature for feature, _ in top_features]
        
        # Criar um DataFrame para o gráfico
        df = pd.DataFrame(index=round_numbers)
        
        # Preencher o DataFrame com as importâncias para cada feature em cada ronda
        for feature in top_feature_names:
            values = []
            for round_num in round_numbers:
                round_str = str(round_num)
                if round_str in feature_importances and feature in feature_importances[round_str]:
                    values.append(feature_importances[round_str][feature])
                else:
                    values.append(0)  # Se não tiver valor, assume zero
            df[feature] = values
        
        # Gerar o gráfico
        plt.figure(figsize=(10, 6))
        for feature in top_feature_names:
            plt.plot(df.index, df[feature], marker='o', label=feature)
        
        plt.title(f'Evolução da Importância das Features ({explainer_type.upper()})')
        plt.xlabel('Ronda')
        plt.ylabel('Importância')
        plt.grid(True)
        plt.xticks(round_numbers)
        plt.legend(title='Features', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'reports/explainability_evolution_{explainer_type}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Gráfico de evolução da explicabilidade {explainer_type.upper()} gerado com sucesso.")

    def explain(self, round_number, training_loss, validation_loss, evaluation_loss):
        """Método principal para gerar todas as explicações e relatórios"""
        print(f"[Client {self.cid}] Gerando explicações...")
        
        # Atualiza as métricas
        self.metrics = {
            'round': round_number,
            'training_loss': training_loss,
            'validation_loss': validation_loss,
            'evaluation_loss': evaluation_loss
        }
        
        # Gera explicações LIME
        print(f"[Client {self.cid}] Gerando explicações LIME...")
        lime_exp = self.explain_lime()
        
        # Gera explicações SHAP
        print(f"[Client {self.cid}] Gerando explicações SHAP...")
        shap_data = self.explain_shap()
        
        # Salva os dados da round
        self.save_round_data(round_number, training_loss, validation_loss, evaluation_loss) 