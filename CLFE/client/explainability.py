import shap
import lime
import lime.lime_tabular
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ModelExplainer:
    def __init__(self, model, feature_names, device="cpu"):
        self.model = model
        self.feature_names = feature_names
        self.device = device
        self.lime_explainer = None
        self.shap_explainer = None
        self.shap_values = None
        
        # Categorias específicas de features
        self.feature_categories = [
            'dl_bitrate',
            'ul_bitrate',
            'cell_x_dl_retx',
            'cell_x_dl_tx',
            'cell_x_ul_retx',
            'cell_x_ul_tx',
            'ul_total_bytes_non_incr',
            'dl_total_bytes_non_incr'
        ]

    def predict_fn(self, x):
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.tensor(x, dtype=torch.float32).to(self.device)
            preds = self.model(x_tensor).cpu().numpy()
            
            # Garantir que retornamos sempre um array 2D com probabilidades para ambas as classes
            # (normal e ataque) - formato requerido para mode="classification"
            if len(preds.shape) == 1:  # se preds for um array 1D
                # Converter para array 2D com probabilidades para ambas as classes
                probs = np.zeros((len(preds), 2))
                probs[:, 1] = preds      # probabilidade de ataque (classe 1)
                probs[:, 0] = 1 - preds  # probabilidade de normal (classe 0)
                return probs
            elif preds.shape[1] == 1:  # se preds for um array 2D mas com apenas 1 coluna
                # Converter para array 2D com probabilidades para ambas as classes
                probs = np.zeros((preds.shape[0], 2))
                probs[:, 1] = preds[:, 0]  # probabilidade de ataque (classe 1)
                probs[:, 0] = 1 - probs[:, 1]  # probabilidade de normal (classe 0)
                return probs
            else:
                # Já está no formato esperado (array 2D com 2 colunas)
                return preds

    def explain_lime(self, X_train, instance):
        if self.lime_explainer is None:
            self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                X_train,
                feature_names=self.feature_names,
                mode="classification"  # Mudando para 'classification' em vez de 'regression'
            )
        # Gerar explicação LIME para a instância
        exp = self.lime_explainer.explain_instance(
            instance, 
            self.predict_fn, 
            num_features=20,  # Aumentado de 10 para 20 para mostrar mais features
            num_samples=5000  # Aumentando o número de amostras para melhorar a precisão
        )
        
        # Personalizar a forma como o gráfico é gerado para evitar truncamento
        # e aplicar o esquema de cores correto (verde para normal, vermelho para ataque)
        return exp
    
    def create_custom_lime_visualization(self, exp, output_path):
        """
        Cria uma visualização LIME personalizada com nomes de features completos
        e esquema de cores correto (verde=normal, vermelho=ataque)
        """
        # Obter explicação como lista
        explanation = exp.as_list()
        
        # Ordenar por importância (valor absoluto do coeficiente)
        explanation.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Separar features e valores
        features = [item[0] for item in explanation]
        values = [item[1] for item in explanation]
        
        # Criar figura maior para acomodar nomes longos
        plt.figure(figsize=(12, 8))
        
        # Definir cores com base no valor e na classe prevista
        # valores NEGATIVOS contribuem para a classe "Normal" (0) - cor VERDE
        # valores POSITIVOS contribuem para a classe "Ataque" (1) - cor VERMELDA
        colors = ['green' if val < 0 else 'red' for val in values]
        
        # Criar gráfico horizontal de barras
        y_pos = range(len(features))
        bars = plt.barh(y_pos, values, color=colors)
        
        # Configurar eixos
        plt.yticks(y_pos, features, fontsize=11)
        plt.xlabel('Contribuição', fontsize=12)
        plt.title('Explicação LIME - Top Features', fontsize=14)
        
        # Adicionar linha zero para referência
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Adicionar uma legenda para clarificar o significado das cores
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Contribui para classificação como Normal'),
            Patch(facecolor='red', label='Contribui para classificação como Ataque')
        ]
        plt.legend(handles=legend_elements, loc='best')
        
        # Ajustar layout para evitar truncamento
        plt.tight_layout()
        
        # Salvar figura
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return output_path

    def explain_shap(self, X_train, n_samples=100):
        if self.shap_explainer is None:
            background = shap.sample(X_train, n_samples, random_state=42) if X_train.shape[0] > n_samples else X_train
            self.shap_explainer = shap.KernelExplainer(self.predict_fn, background)
        self.shap_values = self.shap_explainer.shap_values(X_train)
        return self.shap_values
    
    def _get_feature_type(self, feature_name):
        """Identifica o tipo de uma feature com base em seu nome"""
        # Extrair o número da série temporal (ex: dl_bitrate_42 -> 42)
        parts = feature_name.split('_')
        try:
            time_idx = int(parts[-1]) if parts[-1].isdigit() else None
        except:
            time_idx = None
            
        # Identificar categoria específica da feature
        feature_type = 'other'
        
        # Verificar categorias específicas
        if 'dl_bitrate' in feature_name:
            feature_type = 'dl_bitrate'
        elif 'ul_bitrate' in feature_name:
            feature_type = 'ul_bitrate'
        elif 'cell_' in feature_name and 'dl_retx' in feature_name:
            feature_type = 'cell_x_dl_retx'
        elif 'cell_' in feature_name and 'dl_tx' in feature_name:
            feature_type = 'cell_x_dl_tx'
        elif 'cell_' in feature_name and 'ul_retx' in feature_name:
            feature_type = 'cell_x_ul_retx'
        elif 'cell_' in feature_name and 'ul_tx' in feature_name:
            feature_type = 'cell_x_ul_tx'
        elif 'ul_' in feature_name and 'non_incr' in feature_name:
            feature_type = 'ul_total_bytes_non_incr'
        elif 'dl_' in feature_name and 'non_incr' in feature_name:
            feature_type = 'dl_total_bytes_non_incr'
                
        return feature_type, time_idx
        
    def create_feature_type_importance(self, output_path):
        """
        Gera visualização de importância agregada por tipo de feature, 
        ignorando o número da série temporal
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call explain_shap() first.")
            
        # Agrupar features por tipo
        feature_type_importance = {}
        
        # Calcular a importância absoluta média para cada feature
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Agrupar por tipo de feature
        for idx, feature_name in enumerate(self.feature_names):
            feature_type, _ = self._get_feature_type(feature_name)
            if feature_type not in feature_type_importance:
                feature_type_importance[feature_type] = []
            feature_type_importance[feature_type].append(mean_abs_shap[idx])
        
        # Calcular média por tipo
        aggregated_importance = {
            feature_type: np.mean(values) 
            for feature_type, values in feature_type_importance.items()
            if feature_type in self.feature_categories or (feature_type == 'other' and len(values) > 0)
        }
        
        # Criar visualização
        plt.figure(figsize=(14, 10))
        types = list(aggregated_importance.keys())
        values = list(aggregated_importance.values())
        
        # Ordenar por importância
        sorted_indices = np.argsort(values)
        types = [types[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Plotar o gráfico horizontal de barras
        bars = plt.barh(types, values, color='#1f77b4')
        
        # Adicionar valores nas barras para melhor interpretação
        for i, (bar, value) in enumerate(zip(bars, values)):
            plt.text(
                value + max(values) * 0.01,  # Posição ligeiramente à direita da barra
                bar.get_y() + bar.get_height() / 2,  # Posição no centro vertical da barra
                f'{value:.6f}',  # Formato do valor
                va='center',  # Alinhamento vertical
                fontsize=10
            )
        
        plt.xlabel('Importância Média (SHAP)')
        plt.ylabel('Tipo de Feature')
        plt.title('Importância Média por Tipo de Feature')
        plt.grid(axis='x', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
        
    def create_temporal_trend_plot(self, output_path):
        """
        Gera visualização da evolução temporal da importância 
        das features por série temporal
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call explain_shap() first.")
            
        # Estrutura para armazenar séries temporais
        temporal_series = {}
        
        # Calcular a importância absoluta média para cada feature
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Agrupar por tipo e série temporal
        for idx, feature_name in enumerate(self.feature_names):
            feature_type, time_idx = self._get_feature_type(feature_name)
            
            if time_idx is not None:  # Apenas considerar features com índice temporal
                key = feature_type
                if key not in temporal_series:
                    temporal_series[key] = {}
                    
                if time_idx not in temporal_series[key]:
                    temporal_series[key][time_idx] = []
                    
                temporal_series[key][time_idx].append(mean_abs_shap[idx])
        
        # Calcular média por tipo e tempo
        for key in temporal_series:
            for time_idx in temporal_series[key]:
                temporal_series[key][time_idx] = np.mean(temporal_series[key][time_idx])
        
        # Filtrar apenas categorias específicas
        filtered_series = {k: v for k, v in temporal_series.items() 
                          if k in self.feature_categories}
        
        # Criar visualização com cores distintas para cada categoria - AUMENTAR LARGURA
        plt.figure(figsize=(24, 10))  # Aumentada a largura de 16 para 24
        
        # Definir um conjunto de cores para diferenciar as categorias
        colors = plt.cm.tab10(np.linspace(0, 1, len(filtered_series)))
        
        # Para cada tipo de feature
        for i, (feature_type, series) in enumerate(filtered_series.items()):
            if not series:  # Pular categorias sem dados
                continue
                
            # Ordenar por índice temporal
            times = sorted(series.keys())
            values = [series[t] for t in times]
            
            plt.plot(times, values, marker='o', linestyle='-', 
                    label=feature_type, color=colors[i % len(colors)], 
                    linewidth=2, alpha=0.8, markersize=5)
        
        plt.xlabel('Índice Temporal', fontsize=12)
        plt.ylabel('Importância Média (SHAP)', fontsize=12)
        plt.title('Evolução Temporal da Importância das Features', fontsize=14)
        
        # Mover legenda para baixo do gráfico para dar mais espaço ao conteúdo
        plt.legend(bbox_to_anchor=(0.5, -0.15), loc='upper center', fontsize=10, 
                  ncol=4, frameon=True, fancybox=True, shadow=True)
        
        # Melhorar a densidade dos ticks no eixo x para facilitar a leitura
        xticks_step = max(1, len(set(sum([list(s.keys()) for s in filtered_series.values()], []))) // 20)
        plt.xticks(fontsize=10)
        
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Ajustar para acomodar a legenda em baixo
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return output_path
    
    def create_temporal_heatmap(self, output_path):
        """
        Gera um heatmap temporal da importância das features
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call explain_shap() first.")
            
        # Estrutura para armazenar dados do heatmap
        heatmap_data = {}
        
        # Calcular a importância absoluta média para cada feature
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Agrupar por tipo e série temporal
        for idx, feature_name in enumerate(self.feature_names):
            feature_type, time_idx = self._get_feature_type(feature_name)
            
            if time_idx is not None:  # Apenas considerar features com índice temporal
                if feature_type not in heatmap_data:
                    heatmap_data[feature_type] = {}
                    
                if time_idx not in heatmap_data[feature_type]:
                    heatmap_data[feature_type][time_idx] = []
                    
                heatmap_data[feature_type][time_idx].append(mean_abs_shap[idx])
        
        # Calcular média por tipo e tempo
        for key in heatmap_data:
            for time_idx in heatmap_data[key]:
                heatmap_data[key][time_idx] = np.mean(heatmap_data[key][time_idx])
        
        # Filtrar apenas categorias específicas
        filtered_data = {k: v for k, v in heatmap_data.items() 
                        if k in self.feature_categories}
        
        # Se não houver dados após a filtragem, usar todos os dados
        if not filtered_data:
            filtered_data = heatmap_data
        
        # Encontrar o maior índice temporal em todas as categorias
        max_time_idx = 0
        for category in filtered_data.values():
            if category and max(category.keys()) > max_time_idx:
                max_time_idx = max(category.keys())
        
        # Converter para DataFrame
        heatmap_df = pd.DataFrame({
            feature_type: [
                filtered_data[feature_type].get(t, 0) 
                for t in range(1, max_time_idx + 1)
            ] 
            for feature_type in filtered_data
        })
        heatmap_df.index = range(1, len(heatmap_df) + 1)  # Índices temporais
        
        # Calcular altura dinâmica baseada no número de índices temporais
        # Mínimo de 15 de altura, e no máximo 30
        num_time_indices = len(heatmap_df)
        figure_height = min(max(15, num_time_indices * 0.2), 30)
        
        # Criar visualização com figura mais alta
        plt.figure(figsize=(16, figure_height))
        
        # Aumentar tamanho da fonte para melhor legibilidade
        sns.set(font_scale=1.2)
        
        # Usar uma paleta com maior contraste visual
        heatmap = sns.heatmap(
            heatmap_df, 
            annot=False, 
            fmt=".3f", 
            cmap="viridis", 
            xticklabels=heatmap_df.columns, 
            yticklabels=heatmap_df.index,
            cbar_kws={"shrink": 0.8, "label": "Importância SHAP (abs)"}
        )
        
        # Ajustar tamanho das etiquetas para acomodar nomes longos
        plt.xticks(fontsize=10, rotation=30, ha='right')
        plt.yticks(fontsize=10)
        
        plt.xlabel('Tipo de Feature', fontsize=14)
        plt.ylabel('Índice Temporal', fontsize=14)
        plt.title('Heatmap Temporal da Importância das Features', fontsize=16)
        
        # Adicionar espaço extra para a legenda
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return output_path

    def create_temporal_index_importance(self, output_path):
        """
        Gera visualização da importância agregada pelo número da série temporal,
        independente do tipo de feature.
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call explain_shap() first.")
            
        # Estrutura para armazenar importância por índice temporal
        temporal_index_importance = {}
        
        # Calcular a importância absoluta média para cada feature
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Agrupar por índice temporal, ignorando o tipo de feature
        for idx, feature_name in enumerate(self.feature_names):
            _, time_idx = self._get_feature_type(feature_name)
            
            if time_idx is not None:  # Apenas features com índice temporal
                if time_idx not in temporal_index_importance:
                    temporal_index_importance[time_idx] = []
                    
                temporal_index_importance[time_idx].append(mean_abs_shap[idx])
        
        # Calcular média por índice temporal
        aggregated_importance = {
            time_idx: np.mean(values) 
            for time_idx, values in temporal_index_importance.items()
        }
        
        if not aggregated_importance:
            logger.warning("No temporal indices found in features")
            # Criar um arquivo de placeholder
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Nenhum índice temporal encontrado nas features", 
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.savefig(output_path)
            plt.close()
            return output_path
        
        # Criar visualização
        plt.figure(figsize=(20, 8))
        
        # Ordenar por índice temporal
        time_indices = sorted(aggregated_importance.keys())
        values = [aggregated_importance[t] for t in time_indices]
        
        # Criar gráfico de barras
        plt.bar(time_indices, values, color='#3498db', alpha=0.8, width=0.7)
        
        # Adicionar linha de tendência usando média móvel
        window_size = min(5, len(values))
        if window_size > 1:
            moving_avg = np.convolve(values, np.ones(window_size)/window_size, mode='valid')
            # Calcular índices centralizados para a média móvel
            ma_indices = time_indices[window_size-1:] if window_size % 2 == 0 else time_indices[window_size//2:-(window_size//2)]
            plt.plot(ma_indices, moving_avg, color='#e74c3c', linewidth=2, linestyle='-', label=f'Média Móvel ({window_size} pontos)')
        
        plt.xlabel('Índice Temporal', fontsize=12)
        plt.ylabel('Importância Média (SHAP)', fontsize=12)
        plt.title('Importância do Número da Série Temporal', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # Ajustar eixo x para mostrar mais índices, dependendo da quantidade
        if len(time_indices) > 50:
            plt.xticks(range(min(time_indices), max(time_indices)+1, 5), fontsize=9, rotation=45)
        else:
            plt.xticks(time_indices, fontsize=9)
            
        if window_size > 1:
            plt.legend()
            
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return output_path
    
    def create_timestamp_features_importance(self, output_path):
        """
        Gera visualização da importância das features extraídas do timestamp
        no pré-processamento de dados (hour, dayofweek, day, month, is_weekend).
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not calculated. Call explain_shap() first.")
            
        # Features de timestamp que procuramos
        timestamp_features = ['hour', 'dayofweek', 'day', 'month', 'is_weekend']
        
        # Estrutura para armazenar importância
        timestamp_importance = {feature: [] for feature in timestamp_features}
        
        # Calcular a importância absoluta média para cada feature
        mean_abs_shap = np.abs(self.shap_values).mean(axis=0)
        
        # Identificar features de timestamp
        has_timestamp_features = False
        for idx, feature_name in enumerate(self.feature_names):
            for ts_feature in timestamp_features:
                if feature_name == ts_feature:
                    timestamp_importance[ts_feature].append(mean_abs_shap[idx])
                    has_timestamp_features = True
        
        if not has_timestamp_features:
            logger.warning("No timestamp features found")
            # Criar um arquivo de placeholder
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Nenhuma feature de timestamp encontrada", 
                    horizontalalignment='center', verticalalignment='center', fontsize=14)
            plt.savefig(output_path)
            plt.close()
            return output_path
        
        # Calcular média para cada feature de timestamp
        aggregated_importance = {
            feature: np.mean(values) if values else 0
            for feature, values in timestamp_importance.items()
        }
        
        # Criar visualização
        plt.figure(figsize=(14, 8))
        
        # Preparar dados para o gráfico
        features = list(aggregated_importance.keys())
        values = list(aggregated_importance.values())
        
        # Ordenar por importância
        sorted_indices = np.argsort(values)
        features = [features[i] for i in sorted_indices]
        values = [values[i] for i in sorted_indices]
        
        # Definir cores
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(features)))
        
        # Criar gráfico de barras horizontal
        bars = plt.barh(features, values, color=colors)
        
        # Adicionar valores nas barras
        for i, (bar, value) in enumerate(zip(bars, values)):
            if value > 0:  # Apenas mostrar valores > 0
                plt.text(
                    value + max(values) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f'{value:.6f}',
                    va='center',
                    fontsize=10
                )
        
        plt.xlabel('Importância Média (SHAP)', fontsize=12)
        plt.ylabel('Feature de Timestamp', fontsize=12)
        plt.title('Importância das Features Extraídas do Timestamp', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7, axis='x')
        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()
        
        return output_path
