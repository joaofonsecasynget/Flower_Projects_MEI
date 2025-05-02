"""
Módulo para gerenciamento de metadados de features.
Permite rastrear e documentar todas as features, suas categorias, e valores.
"""
import os
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd

# Configurar logger
logger = logging.getLogger("explainability.feature_metadata")

class FeatureMetadata:
    """
    Classe para gerenciar metadados de features, incluindo categorias, 
    normalização, e se são usadas no treinamento.
    """
    
    def __init__(self, metadata_path: Optional[str] = None):
        """
        Inicializa o gerenciador de metadados.
        
        Args:
            metadata_path: Caminho para carregar/salvar o arquivo de metadados.
                           Se None, usa um caminho padrão.
        """
        self.metadata_path = metadata_path or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "../data/feature_metadata.json"
        )
        self.features = {}
        self.categories = {}
        self.loaded = False
        
        # Tentar carregar metadados existentes
        try:
            self.load()
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info(f"Nenhum arquivo de metadados encontrado em {self.metadata_path} ou formato inválido")
            self.loaded = False

    def load(self) -> bool:
        """
        Carrega metadados de features do arquivo JSON.
        
        Returns:
            True se sucesso, False caso contrário
        """
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r') as f:
                    data = json.load(f)
                    self.features = data.get('features', {})
                    self.categories = data.get('categories', {})
                    self.loaded = True
                    logger.info(f"Metadados carregados: {len(self.features)} features, {len(self.categories)} categorias")
                    return True
        except Exception as e:
            logger.error(f"Erro ao carregar metadados: {str(e)}")
        return False
    
    def save(self) -> bool:
        """
        Salva metadados de features para arquivo JSON.
        
        Returns:
            True se sucesso, False caso contrário
        """
        try:
            # Garantir que o diretório existe
            os.makedirs(os.path.dirname(self.metadata_path), exist_ok=True)
            
            with open(self.metadata_path, 'w') as f:
                json.dump({
                    'features': self.features,
                    'categories': self.categories
                }, f, indent=2)
            logger.info(f"Metadados salvos em {self.metadata_path}")
            return True
        except Exception as e:
            logger.error(f"Erro ao salvar metadados: {str(e)}")
            return False
    
    def extract_metadata_from_dataset(
        self, 
        df: pd.DataFrame, 
        normalize: bool = False,
        remove_identifiers: bool = True,
        extract_time_features: bool = True
    ) -> Dict[str, Any]:
        """
        Extrai metadados automaticamente de um DataFrame.
        
        Args:
            df: DataFrame para extrair metadados
            normalize: Se deve calcular estatísticas de normalização
            remove_identifiers: Se deve marcar identificadores para remoção
            extract_time_features: Se deve extrair features a partir de colunas temporais
            
        Returns:
            Dicionário com metadados extraídos
        """
        # Identificadores conhecidos
        identifiers = ['indice', 'índice', 'index', 'imeisv', '_id']
        
        # Categorias conhecidas e seus padrões
        known_categories = {
            'dl_bitrate': ['dl_bitrate'],
            'ul_bitrate': ['ul_bitrate'],
            'cell_x_dl_retx': ['cell_x_dl_retx'],
            'cell_x_dl_tx': ['cell_x_dl_tx'],
            'cell_x_ul_retx': ['cell_x_ul_retx'],
            'cell_x_ul_tx': ['cell_x_ul_tx']
        }
        
        # Inicializar categorias vazias
        categories = {cat: [] for cat in known_categories.keys()}
        categories['identifier'] = []
        categories['time_features'] = []
        categories['other'] = []
        
        # Processar cada coluna
        for col in df.columns:
            feature_info = {
                'type': str(df[col].dtype),
                'used_in_training': True,  # Default
                'category': 'other'  # Default
            }
            
            # Verificar identificadores
            if any(id_pattern in str(col).lower() for id_pattern in identifiers):
                feature_info['category'] = 'identifier'
                feature_info['used_in_training'] = not remove_identifiers
                categories['identifier'].append(col)
                continue
            
            # Verificar categorias conhecidas
            category_found = False
            for category, patterns in known_categories.items():
                if any(pattern in str(col) for pattern in patterns):
                    feature_info['category'] = category
                    categories[category].append(col)
                    category_found = True
                    break
            
            # Se não encontrou categoria, vai para 'other'
            if not category_found:
                categories['other'].append(col)
            
            # Calcular estatísticas de normalização
            if normalize and pd.api.types.is_numeric_dtype(df[col]):
                feature_info['normalization'] = {
                    'mean': float(df[col].mean()),
                    'std': float(df[col].std()),
                    'min': float(df[col].min()),
                    'max': float(df[col].max())
                }
            
            # Adicionar aos metadados
            self.features[col] = feature_info
        
        # Features temporais potenciais
        if extract_time_features and '_time' in df.columns:
            # Marcar _time como identificador
            if '_time' in self.features:
                self.features['_time']['category'] = 'identifier'
                self.features['_time']['used_in_training'] = False
                
                if 'other' in categories and '_time' in categories['other']:
                    categories['other'].remove('_time')
                
                if '_time' not in categories['identifier']:
                    categories['identifier'].append('_time')
            
            # Adicionar features derivadas de _time
            time_features = ['hour', 'dayofweek', 'day', 'month', 'is_weekend']
            for feat in time_features:
                self.features[feat] = {
                    'type': 'numeric',
                    'category': 'time_features',
                    'used_in_training': True,
                    'derived_from': '_time'
                }
                categories['time_features'].append(feat)
        
        # Atualizar as categorias
        self.categories = categories
        
        return self.features
    
    def normalize_value(self, feature_name: str, value: float) -> float:
        """
        Normaliza um valor usando as estatísticas armazenadas.
        
        Args:
            feature_name: Nome da feature
            value: Valor a ser normalizado
            
        Returns:
            Valor normalizado
        """
        if feature_name not in self.features:
            return value
        
        feature_info = self.features[feature_name]
        if 'normalization' not in feature_info:
            return value
        
        norm = feature_info['normalization']
        if norm['std'] == 0:
            return 0  # Evitar divisão por zero
        
        return (value - norm['mean']) / norm['std']
    
    def denormalize_value(self, feature_name: str, normalized_value: float) -> float:
        """
        Converte um valor normalizado de volta para a escala original.
        
        Args:
            feature_name: Nome da feature
            normalized_value: Valor normalizado
            
        Returns:
            Valor na escala original
        """
        if feature_name not in self.features:
            return normalized_value
        
        feature_info = self.features[feature_name]
        if 'normalization' not in feature_info:
            return normalized_value
        
        norm = feature_info['normalization']
        
        # Verificar se o valor já parece estar na escala original
        # Isso evita denormalização múltipla, que pode causar valores extremamente altos
        if abs(normalized_value) > 0 and abs(normalized_value) > 0.1 * norm['max']:
            logger.warning(f"Valor {normalized_value} para feature {feature_name} pode já estar denormalizado. Retornando sem modificar.")
            return normalized_value
        
        return normalized_value * norm['std'] + norm['mean']
    
    def get_category(self, feature_name: str) -> str:
        """
        Retorna a categoria de uma feature.
        
        Args:
            feature_name: Nome da feature
            
        Returns:
            Nome da categoria
        """
        if feature_name in self.features:
            return self.features[feature_name].get('category', 'other')
        
        # Tenta determinar a categoria com base em padrões no nome
        feature_str = str(feature_name)
        
        # Padrões para features temporais
        time_patterns = ['hour', 'day', 'month', 'year', 'weekday', 'minute', 'second', 'is_weekend']
        for pattern in time_patterns:
            if pattern in feature_str.lower():
                return 'time_features'
        
        # Padrões para identificadores
        identifiers = ['indice', 'índice', 'index', 'imeisv']
        for id_pattern in identifiers:
            if id_pattern in feature_str.lower():
                return 'identifier'
        
        # Categorias específicas
        for category_name, features in self.categories.items():
            # Se a feature estiver explicitamente listada nesta categoria
            if feature_name in features:
                return category_name
            
            # Tenta casar padrões para categorias conhecidas
            if category_name in ['dl_bitrate', 'ul_bitrate', 'cell_x_dl_retx', 'cell_x_dl_tx', 'cell_x_ul_retx', 'cell_x_ul_tx']:
                if category_name in feature_str:
                    return category_name
        
        # Fallback para 'other'
        return 'other'
    
    def get_features_by_category(self, category: str) -> List[str]:
        """
        Retorna todas as features de uma categoria específica.
        
        Args:
            category: Nome da categoria
            
        Returns:
            Lista de nomes de features
        """
        return self.categories.get(category, [])
    
    def get_training_features(self) -> List[str]:
        """
        Retorna todas as features marcadas para uso no treinamento.
        
        Returns:
            Lista de nomes de features
        """
        return [
            name for name, info in self.features.items() 
            if info.get('used_in_training', True)
        ]
    
    def mark_for_training(self, feature_name: str, use_in_training: bool = True) -> bool:
        """
        Marca uma feature para uso ou não no treinamento.
        
        Args:
            feature_name: Nome da feature
            use_in_training: Se deve ser usada no treinamento
            
        Returns:
            True se a operação foi bem-sucedida
        """
        if feature_name in self.features:
            self.features[feature_name]['used_in_training'] = use_in_training
            return True
        return False
    
    def export_summary(self, output_path: Optional[str] = None) -> str:
        """
        Exporta um resumo dos metadados das features em formato texto.
        
        Args:
            output_path: Caminho para salvar o resumo. Se None, retorna a string.
            
        Returns:
            String com o resumo se output_path for None, senão o caminho do arquivo
        """
        summary = "RESUMO DE FEATURES\n"
        summary += "================\n\n"
        
        # Total por categoria
        summary += "Contagem de Features por Categoria:\n"
        for category, features in self.categories.items():
            num_training = sum(
                1 for f in features 
                if f in self.features and self.features[f].get('used_in_training', True)
            )
            summary += f"- {category}: {len(features)} features ({num_training} usadas no treinamento)\n"
        
        # Features derivadas
        derived_features = [
            name for name, info in self.features.items()
            if 'derived_from' in info
        ]
        if derived_features:
            summary += "\nFeatures Derivadas:\n"
            for feat in derived_features:
                source = self.features[feat].get('derived_from', 'desconhecido')
                summary += f"- {feat} (derivada de: {source})\n"
        
        # Se solicitado, salvar em arquivo
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w') as f:
                f.write(summary)
            return output_path
        
        return summary
    
    def align_datasets(
        self, 
        df_original: pd.DataFrame, 
        df_client: pd.DataFrame, 
        feature_names: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """
        Alinha dois DataFrames para ter as mesmas features e ordem.
        
        Args:
            df_original: DataFrame original
            df_client: DataFrame do cliente
            feature_names: Lista opcional de nomes das features para padronizar
            
        Returns:
            Tupla (df_original_alinhado, df_client_alinhado, lista_features)
        """
        # Se lista de features não for fornecida, usar todas as features de treinamento
        if not feature_names:
            feature_names = self.get_training_features()
        
        # Features presentes em ambos os DataFrames
        common_features = [f for f in feature_names if f in df_original.columns or f in df_client.columns]
        
        # Features derivadas que podem precisar ser geradas
        derived_features = {
            name: info.get('derived_from') 
            for name, info in self.features.items()
            if 'derived_from' in info and name in common_features
        }
        
        # Processar features temporais no DataFrame original
        if '_time' in df_original.columns and derived_features:
            time_derived = [feat for feat, source in derived_features.items() if source == '_time']
            if time_derived:
                # Certifique-se de que _time está no formato datetime
                if not pd.api.types.is_datetime64_dtype(df_original['_time']):
                    try:
                        df_original['_time'] = pd.to_datetime(df_original['_time'])
                    except:
                        logger.warning("Não foi possível converter _time para datetime no DataFrame original")
                
                # Extrair features temporais
                if pd.api.types.is_datetime64_dtype(df_original['_time']):
                    for feat in time_derived:
                        if feat == 'hour' and feat not in df_original.columns:
                            df_original[feat] = df_original['_time'].dt.hour
                        elif feat == 'dayofweek' and feat not in df_original.columns:
                            df_original[feat] = df_original['_time'].dt.dayofweek
                        elif feat == 'day' and feat not in df_original.columns:
                            df_original[feat] = df_original['_time'].dt.day
                        elif feat == 'month' and feat not in df_original.columns:
                            df_original[feat] = df_original['_time'].dt.month
                        elif feat == 'is_weekend' and feat not in df_original.columns:
                            df_original[feat] = (df_original['_time'].dt.dayofweek >= 5).astype(int)
        
        # Criar listas de features alinhadas 
        all_features = []
        for feat in common_features:
            # Adicionar à lista de features
            all_features.append(feat)
            
            # Adicionar a coluna se estiver faltando em algum DataFrame
            for df, df_name in [(df_original, 'original'), (df_client, 'client')]:
                if feat not in df.columns:
                    logger.info(f"Adicionando coluna {feat} faltante ao DataFrame {df_name}")
                    df[feat] = 0.0  # Valor padrão para dados faltantes
        
        # Verificar se precisamos adicionar colunas extras ao DataFrame original para corresponder ao cliente
        # Isso é útil para o caso de features derivadas durante treinamento mas não presentes no original
        if len(df_original.columns) < len(df_client.columns):
            missing_count = len(df_client.columns) - len(df_original.columns)
            for i in range(missing_count):
                feat_name = f"extra_feature_{i}"
                if feat_name not in all_features:
                    all_features.append(feat_name)
                    df_original[feat_name] = 0.0
                    logger.info(f"Adicionada feature {feat_name} ao DataFrame original para alinhamento")
        
        # Retornar DataFrames filtrados e alinhados + lista final de features
        return df_original[all_features], df_client[all_features], all_features


# Criar instância singleton para uso em todo o código
feature_metadata = FeatureMetadata()
