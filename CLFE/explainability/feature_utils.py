"""
Funções para extração e mapeamento de features para explicabilidade.
"""
import os
import re
import logging
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

# Configurar logger
logger = logging.getLogger("explainability.feature_utils")

def get_feature_category(feature_name):
    """Determina a categoria de uma feature com base em seu nome"""
    if "dl_bitrate" in feature_name:
        return "dl_bitrate"
    elif "ul_bitrate" in feature_name:
        return "ul_bitrate"
    elif "cell_x_dl_retx" in feature_name:
        return "cell_x_dl_retx"
    elif "cell_x_dl_tx" in feature_name:
        return "cell_x_dl_tx"
    elif "cell_x_ul_retx" in feature_name:
        return "cell_x_ul_retx"
    elif "cell_x_ul_tx" in feature_name:
        return "cell_x_ul_tx"
    elif "ul_total_bytes_non_incr" in feature_name:
        return "ul_total_bytes_non_incr"
    elif "dl_total_bytes_non_incr" in feature_name:
        return "dl_total_bytes_non_incr"
    elif feature_name in ["hour", "dayofweek", "day", "month", "is_weekend"]:
        return "timestamp"
    else:
        return "other"

def extract_feature_names(client_id, num_features, instance=None, dataset_path=None, is_original_dataset=False, feature_columns=None):
    """
    Extrai nomes reais das features usando múltiplas estratégias:
    1. Usar nomes de colunas fornecidos diretamente (se disponíveis)
    2. Tentar extrair do arquivo LIME HTML
    3. Tentar extrair do arquivo de explicação LIME
    4. Inferir a partir dos valores da instância
    5. Usar nomes genéricos com categorias específicas do domínio

    Args:
        client_id: ID do cliente
        num_features: Número total de features
        instance: Instância para a qual queremos explicação
        dataset_path: Caminho opcional para o dataset original (para extrair nomes de colunas)
        is_original_dataset: Indica se estamos usando o dataset original
        feature_columns: Lista de nomes de colunas já extraídos do dataset original

    Returns:
        Tuple[List[str], Dict[str, str]]: (nomes das features, mapping de nomes)
    """
    
    logger.info(f"Extraindo nomes reais das features para {num_features} features")
    
    # Se temos os nomes das colunas diretamente, usá-los
    if feature_columns and len(feature_columns) > 0:
        logger.info(f"Usando nomes de features já extraídos do dataset original")
        
        # Se o número de features não bater, precisamos ajustar
        if len(feature_columns) != num_features:
            logger.warning(f"Incompatibilidade no número de features: temos {len(feature_columns)} nomes, esperados {num_features}")
            
            # Se temos mais colunas no modelo, precisamos adicionar colunas extras
            if len(feature_columns) < num_features:
                extra_features = num_features - len(feature_columns)
                logger.info(f"Adicionando {extra_features} nomes de features extras")
                
                # Se a diferença for 5, assumimos que são as features temporais
                if extra_features == 5:
                    temporal_features = ['hour', 'dayofweek', 'day', 'month', 'is_weekend']
                    for feature_name in temporal_features:
                        feature_columns.append(feature_name)
                    logger.info(f"Adicionadas features temporais: {temporal_features}")
                else:
                    # Caso contrário, usamos nomes genéricos
                    for i in range(extra_features):
                        feature_columns.append(f"extra_feature_{i}")
            else:
                # Se temos menos colunas no modelo, truncar
                logger.warning(f"Truncando lista de features para {num_features} colunas")
                feature_columns = feature_columns[:num_features]
        
        # Mapeamento de nomes
        feature_map = {f"feature_{i}": name for i, name in enumerate(feature_columns)}
        
        # Garantir que as últimas 5 colunas sejam as features temporais quando temos exatamente 5 features extras
        # Isso é crítico para o caso do cliente que adiciona 5 features temporais ao dataset original
        client_has_temporal_features = False
        if num_features - len(feature_columns) + 5 == num_features and is_original_dataset:
            # Verificar se as últimas 5 features têm nomes como "extra_feature_X"
            last_five_features = feature_columns[-5:]
            generic_names = [f for f in last_five_features if f.startswith("extra_feature_")]
            
            # Se todas as 5 últimas features têm nomes genéricos, substituí-las pelas features temporais
            if len(generic_names) == 5:
                temporal_features = ['hour', 'dayofweek', 'day', 'month', 'is_weekend']
                for i in range(5):
                    idx = len(feature_columns) - 5 + i
                    if idx < len(feature_columns):
                        feature_columns[idx] = temporal_features[i]
                        feature_map[f"feature_{idx}"] = temporal_features[i]
                
                logger.info(f"Substituídas últimas 5 features genéricas pelas features temporais: {temporal_features}")
                client_has_temporal_features = True
        
        # Verificar também se já temos features temporais no dataset (mesmo sem diferença de tamanho)
        if not client_has_temporal_features:
            temporal_features = ['hour', 'dayofweek', 'day', 'month', 'is_weekend']
            temporal_in_data = [f for f in feature_columns if f in temporal_features]
            if temporal_in_data:
                logger.info(f"Encontradas features temporais no dataset: {temporal_in_data}")
        
        logger.info(f"Usando {len(feature_columns)} nomes específicos de features do dataset original")
        logger.info(f"Exemplos de nomes: {feature_columns[:5]}, ...")
        
        return feature_columns, feature_map
    
    # Se estamos usando o dataset original e temos o caminho, tentar extrair nomes das colunas
    if is_original_dataset and dataset_path and os.path.exists(dataset_path):
        try:
            import pandas as pd
            logger.info(f"Obtendo nomes de features diretamente do dataset original: {dataset_path}")
            df = pd.read_csv(dataset_path)
            
            # Remover colunas irrelevantes/não-numéricas e target
            irrelevant_cols = ['índice', 'indice', 'imeisv', 'index', '_time']  # Colunas a remover
            target_col = df.columns[-1]  # Assumindo que a última coluna é o target
            cols_to_drop = [col for col in irrelevant_cols if col in df.columns] + [target_col]
            df_features = df.drop(columns=cols_to_drop)
            
            # Obter nomes de colunas numéricas
            numeric_cols = df_features.select_dtypes(include=['number']).columns.tolist()
            
            # Se o número de features não bater, precisamos ajustar
            if len(numeric_cols) != num_features:
                logger.warning(f"Incompatibilidade no número de features: dataset tem {len(numeric_cols)}, modelo espera {num_features}")
                
                # Se temos mais colunas no modelo, precisamos adicionar colunas extras
                if len(numeric_cols) < num_features:
                    extra_features = num_features - len(numeric_cols)
                    logger.info(f"Adicionando {extra_features} nomes de features extras")
                    for i in range(extra_features):
                        numeric_cols.append(f"extra_feature_{i}")
                else:
                    # Se temos menos colunas no modelo, truncar
                    logger.warning(f"Truncando lista de features para {num_features} colunas")
                    numeric_cols = numeric_cols[:num_features]
            
            # Mapeamento de nomes
            feature_map = {f"feature_{i}": name for i, name in enumerate(numeric_cols)}
            
            logger.info(f"Identificados {len(numeric_cols)} nomes específicos de features do dataset original ({100*len(numeric_cols)/num_features:.1f}%)")
            logger.info(f"Exemplos de nomes: {numeric_cols[:5]}, ...")
            
            return numeric_cols, feature_map
            
        except Exception as e:
            logger.error(f"Erro ao extrair nomes do dataset original: {e}")
            # Continuar com as outras estratégias se falhar
    
    # Dicionário para mapear índices de features aos nomes reais
    feature_map = {}
    
    # 1. Tentar extrair nomes de features do arquivo HTML do LIME
    lime_html_path = f"client/reports/client_{client_id}/lime_final.html"
    logger.info(f"Tentando extrair nomes de features de {lime_html_path}")
    
    if os.path.exists(lime_html_path):
        try:
            with open(lime_html_path, 'r', encoding='utf-8') as file:
                html_content = file.read()
                soup = BeautifulSoup(html_content, 'html.parser')
                # Procurar tabelas com nomes de features
                feature_names_found = []
                for table in soup.find_all('table'):
                    for row in table.find_all('tr'):
                        cells = row.find_all('td')
                        if len(cells) >= 2:  # Pelo menos 2 células por linha
                            feature_text = cells[0].get_text().strip()
                            if "<=" in feature_text or ">" in feature_text:
                                parts = re.split(r'[<=>]', feature_text)
                                if len(parts) > 0:
                                    feature_name = parts[0].strip()
                                    feature_names_found.append(feature_name)
                
                if feature_names_found:
                    logger.info(f"Encontrados {len(feature_names_found)} possíveis nomes de features no HTML")
                    # Adicionar ao dicionário de mapeamento
                    for i, name in enumerate(feature_names_found[:num_features]):
                        feature_map[i] = name
        except Exception as e:
            logger.warning(f"Erro ao extrair nomes de features do HTML: {e}")
    
    # 2. Buscar em arquivos de texto (explicações)
    explanation_txt_path = f"client/reports/client_{client_id}/lime_explanation.txt"
    logger.info(f"Buscando nomes de features em {explanation_txt_path}")
    
    if os.path.exists(explanation_txt_path):
        try:
            with open(explanation_txt_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
                pattern = r'^([a-zA-Z0-9_]+)[\s:]+'
                for line in lines:
                    match = re.match(pattern, line.strip())
                    if match:
                        feature_name = match.group(1)
                        if feature_name not in feature_map.values():
                            # Adicionar ao mapeamento
                            next_index = len(feature_map)
                            if next_index < num_features:
                                feature_map[next_index] = feature_name
        except Exception as e:
            logger.warning(f"Erro ao extrair nomes de features do TXT: {e}")
    
    # 3. Inferir nomes com base nas convenções do domínio
    known_features = {
        'hour': None, 'dayofweek': None, 'day': None, 'month': None, 'is_weekend': None,
        'dl_bitrate': [], 'ul_bitrate': [], 
        'cell_x_dl_retx': [], 'cell_x_dl_tx': [], 
        'cell_x_ul_retx': [], 'cell_x_ul_tx': [],
        'dl_total_bytes_non_incr': [], 'ul_total_bytes_non_incr': []
    }
    
    # 4. Análise de valor da instância atual para identificar padrões
    if instance is not None:
        # Identificar padrões nos valores
        if len(instance) == num_features:
            # Análise de timestamp
            timestamp_indices = []
            for i in range(min(10, len(instance))):
                # Os primeiros valores geralmente são relacionados ao timestamp
                if 0 <= instance[i] <= 24:  # Possível hora
                    timestamp_indices.append(i)
                    if i < 5 and i not in feature_map:
                        feature_map[i] = list(known_features.keys())[i]
            
            # Identificar segmentos por padrões de valor
            segments = []
            current_segment = []
            prev_val = None
            
            for i in range(len(instance)):
                if i in timestamp_indices:
                    continue
                
                val = instance[i]
                
                # Iniciar novo segmento se houver mudança significativa no padrão
                if prev_val is not None:
                    if (val == 0 and prev_val != 0) or (val != 0 and prev_val == 0):
                        if current_segment:
                            segments.append(current_segment)
                            current_segment = []
                
                current_segment.append(i)
                prev_val = val
            
            if current_segment:
                segments.append(current_segment)
            
            # Mapear segmentos para categorias
            if segments:
                feature_categories = list(known_features.keys())[5:]  # Pular timestamp
                segment_size = len(segments)
                category_size = min(segment_size, len(feature_categories))
                
                # Mapear cada segmento para uma categoria
                for idx, segment in enumerate(segments[:category_size]):
                    category = feature_categories[idx]
                    category_count = len(segment)
                    
                    # Nomear features no segmento
                    for i, feature_idx in enumerate(segment):
                        if feature_idx not in feature_map:
                            feature_map[feature_idx] = f"{category}_{i}"
    
    # 5. Preencher os indices restantes com nomes genéricos
    for i in range(num_features):
        if i not in feature_map:
            feature_map[i] = f"feature_{i}"
    
    # Construir a lista ordenada de nomes de features
    feature_names = [feature_map[i] for i in range(num_features)]
    
    # Log de informações para diagnóstico
    specific_features = [name for name in feature_names if not name.startswith('feature_')]
    logger.info(f"Identificados {len(specific_features)} nomes específicos de features ({(len(specific_features)/num_features*100):.1f}%)")
    logger.info(f"Exemplos de nomes: {specific_features[:5]}, ...")
    
    return feature_names, feature_map
