"""
Funções para carregamento de modelos e dados.
"""
import os
import sys
import logging
import numpy as np
import torch
import pandas as pd
import time

# Importar o sistema de metadados de features
from .feature_metadata import feature_metadata

# Configurar logger
logger = logging.getLogger("explainability.data_loader")

def load_train_data(client_id, dataset_path=None):
    """
    Carrega os dados de treinamento:
    - Se dataset_path for fornecido, carrega o dataset original completo
    - Caso contrário, carrega apenas os dados de treinamento do cliente
    """
    if dataset_path and os.path.exists(dataset_path):
        logger.info(f"Carregando Dataset original completo de {dataset_path}")
        
        # Importar pandas para ler o CSV
        import pandas as pd
        
        # Carregar o dataset original
        df = pd.read_csv(dataset_path)
        
        # Se existir metadados de features, usá-los para categorização
        metadata_path = os.path.join(os.path.dirname(dataset_path), 'feature_metadata.json')
        if os.path.exists(metadata_path):
            feature_metadata.metadata_path = metadata_path
            feature_metadata.load()
            logger.info(f"Metadados de features carregados de {metadata_path}")
        else:
            # Se não existirem metadados, extraí-los agora e salvá-los
            logger.info("Gerando metadados de features a partir do dataset")
            feature_metadata.extract_metadata_from_dataset(df, normalize=True)
            os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
            feature_metadata.metadata_path = metadata_path
            feature_metadata.save()
            logger.info(f"Metadados de features salvos em {metadata_path}")
            
        # Converter a coluna _time para datetime logo no início do processamento
        # Isso vai garantir que possamos usar essa coluna para calcular features temporais
        if '_time' in df.columns:
            try:
                # Tentar múltiplos formatos para ser mais robusto
                try:
                    # Método 1: Converter diretamente (padrão)
                    df['_time'] = pd.to_datetime(df['_time'])
                    logger.info("Coluna _time convertida para datetime com formato padrão")
                except:
                    # Método 2: Tentar com formato 'mixed' (auto-detecção)
                    try:
                        df['_time'] = pd.to_datetime(df['_time'], format='mixed')
                        logger.info("Coluna _time convertida para datetime com formato 'mixed'")
                    except:
                        # Método 3: Forçar ISO8601
                        try:
                            df['_time'] = pd.to_datetime(df['_time'], format='ISO8601')
                            logger.info("Coluna _time convertida para datetime com formato 'ISO8601'")
                        except Exception as e:
                            # Método 4: Tentativa adicional com utc=True
                            df['_time'] = pd.to_datetime(df['_time'], utc=True, errors='coerce')
                            logger.info("Coluna _time convertida para datetime com utc=True e erros corrigidos")
                
                # Verificar quantos valores foram convertidos com sucesso
                null_times = df['_time'].isna().sum()
                if null_times > 0:
                    logger.warning(f"{null_times} valores de _time não puderam ser convertidos para datetime")
                
            except Exception as e:
                logger.warning(f"Erro ao converter _time para datetime: {str(e)}")
        
        # Tratar identificadores e target
        target_col = 'attack'
        identifiers = ['indice', 'imeisv', 'index']
        
        # Preservar os índices originais para referência
        original_indices = None
        for id_col in identifiers:
            if id_col in df.columns:
                original_indices = df[id_col].values
                break
        
        # Separar o target se existir
        target_values = None
        if target_col in df.columns:
            target_values = df[target_col].values
            df = df.drop(columns=[target_col])
        
        # Processar features temporais se _time estiver presente
        # Importante: fazer isso ANTES de remover colunas
        if '_time' in df.columns:
            logger.info("Extraindo features temporais de _time")
            try:
                # Verificar quais features temporais já estão presentes
                time_features = feature_metadata.get_features_by_category('time_features')
                
                # Criar features temporais faltantes
                logger.info("Calculando valores reais para features temporais")
                df['hour'] = df['_time'].dt.hour
                df['dayofweek'] = df['_time'].dt.dayofweek
                df['day'] = df['_time'].dt.day
                df['month'] = df['_time'].dt.month
                df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
                
                # Registrar nos metadados se for necessário
                for feature in ['hour', 'dayofweek', 'day', 'month', 'is_weekend']:
                    if feature not in feature_metadata.features:
                        logger.info(f"Adicionando feature temporal {feature} aos metadados")
                        feature_metadata.features[feature] = {
                            'type': 'numeric',
                            'category': 'time_features',
                            'used_in_training': True,
                            'derived_from': '_time'
                        }
                
                # Remover _time após extração
                logger.info("Features temporais calculadas, removendo coluna _time")
                df = df.drop(columns=['_time'])
            except Exception as e:
                logger.warning(f"Erro ao processar features temporais: {str(e)}")
        
        # Obter features a serem usadas no treinamento (excluir identificadores)
        training_features = feature_metadata.get_training_features()
        
        # Remover colunas que não devem ser usadas no treinamento
        cols_to_remove = [col for col in df.columns if col not in training_features]
        if cols_to_remove:
            logger.info(f"Removendo colunas não usadas no treinamento: {cols_to_remove}")
            df = df.drop(columns=cols_to_remove)
        
        # Preservar os valores originais para reporting
        X_original_values = df.values.copy()
        
        # Converter colunas para tipo numérico (necessário para o modelo)
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                try:
                    df[col] = pd.to_numeric(df[col])
                except:
                    logger.warning(f"Não foi possível converter coluna {col} para numérico, usando zeros")
                    df[col] = 0
        
        # Features para usar
        feature_columns = list(df.columns)
        
        # Normalizar os dados
        X_original = df.values
        
        # Carregar dados do cliente se disponíveis
        client_data_path = f"client/reports/client_{client_id}/X_train.npy"
        try:
            X_train_client = np.load(client_data_path)
            logger.info(f"Dados de treinamento do cliente {client_id} carregados de {client_data_path}")
        except:
            logger.warning(f"Não foi possível carregar dados do cliente {client_id}, usando dataset original")
            return X_original, True, original_indices, target_values, feature_columns, X_original_values
        
        logger.info(f"Dataset original tem {len(X_original)} instâncias, cliente {client_id} tem {len(X_train_client)} instâncias")
        
        # Verificar se os formatos são compatíveis
        missing_features = 0
        if X_original.shape[1] != X_train_client.shape[1]:
            if X_original.shape[1] < X_train_client.shape[1]:
                missing_features = X_train_client.shape[1] - X_original.shape[1]
                logger.warning(f"Incompatibilidade de features: dataset original tem {X_original.shape[1]} features, dados do cliente têm {X_train_client.shape[1]} features")
                
                # Se a diferença for exatamente 5, provavelmente são as features temporais
                if missing_features == 5 and '_time' in df.columns:
                    # Converter para datetime se ainda não for
                    if not pd.api.types.is_datetime64_dtype(df['_time']):
                        try:
                            df['_time'] = pd.to_datetime(df['_time'])
                        except:
                            logger.warning("Não foi possível converter _time para datetime")
                            
                    # Calcular as features temporais diretamente
                    if pd.api.types.is_datetime64_dtype(df['_time']):
                        logger.info("Gerando features temporais a partir de _time para alinhamento")
                        df['hour'] = df['_time'].dt.hour
                        df['dayofweek'] = df['_time'].dt.dayofweek
                        df['day'] = df['_time'].dt.day
                        df['month'] = df['_time'].dt.month
                        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
                        
                        # Também adicionar ao X_original_values para manter coerência
                        original_values_with_time = np.hstack((X_original_values, 
                                                             df[['hour', 'dayofweek', 'day', 'month', 'is_weekend']].values))
                        X_original_values = original_values_with_time
                    else:
                        # Fallback: adicionar colunas extras com zeros
                        logger.warning("Não foi possível gerar features temporais, adicionando colunas com zeros")
                        padding = np.zeros((X_original.shape[0], missing_features), dtype=np.float32)
                        X_original = np.hstack((X_original, padding))
                        
                        # Também adicionar ao X_original_values para manter coerência
                        padding_values = np.zeros((X_original_values.shape[0], missing_features), dtype=np.float32)
                        X_original_values = np.hstack((X_original_values, padding_values))
                    
                    # Adicionar nomes das features temporais
                    for feature in ['hour', 'dayofweek', 'day', 'month', 'is_weekend']:
                        if feature not in feature_columns:
                            feature_columns.append(feature)
                else:
                    # Caso genérico: adicionar colunas extras com zeros
                    padding = np.zeros((X_original.shape[0], missing_features), dtype=np.float32)
                    X_original = np.hstack((X_original, padding))
                    
                    # Também adicionar ao X_original_values para manter coerência
                    padding_values = np.zeros((X_original_values.shape[0], missing_features), dtype=np.float32)
                    X_original_values = np.hstack((X_original_values, padding_values))
                    
                    # Adicionar nomes genéricos para as features extras
                    for i in range(missing_features):
                        feature_columns.append(f"extra_feature_{i}")
                
                logger.info(f"Dataset original agora tem {X_original.shape[1]} features, compatível com os dados do cliente")
        
        # Força usar nomes reais de features temporais se o número de features extras for 5
        # Este é o caso típico onde o cliente adiciona 5 features temporais (hour, dayofweek, day, month, is_weekend)
        if missing_features == 5 and len(feature_columns) >= X_original.shape[1] - 5:
            temporal_features = ['hour', 'dayofweek', 'day', 'month', 'is_weekend']
            for i in range(5):
                idx = len(feature_columns) - 5 + i
                if idx < len(feature_columns):
                    old_name = feature_columns[idx]
                    if 'extra_feature' in old_name:
                        feature_columns[idx] = temporal_features[i]
                        logger.info(f"Substituindo {old_name} por {temporal_features[i]}")
        
        logger.info(f"Dataset original agora tem {X_original.shape[1]} features, compatível com os dados do cliente")
        
        # Converter para float32 para garantir compatibilidade com o PyTorch
        X_original = X_original.astype(np.float32)
        
        # Retornar o dataset original, flag indicando uso do dataset original, índices originais, valores do target e nomes das colunas
        return X_original, True, original_indices, target_values, feature_columns, X_original_values
    
    else:
        # Tentar carregar dados de treinamento do cliente
        try:
            X_train_client = np.load(f"client/reports/client_{client_id}/X_train.npy")
            y_train_client = np.load(f"client/reports/client_{client_id}/y_train.npy")
            logger.info(f"Dados de treinamento do cliente {client_id} carregados")
            
            # Tentar carregar nomes das features
            try:
                # Verificar se há um arquivo de metadados de features
                metadata_path = f"client/reports/client_{client_id}/feature_metadata.json"
                if os.path.exists(metadata_path):
                    feature_metadata.metadata_path = metadata_path
                    feature_metadata.load()
                    feature_columns = feature_metadata.get_training_features()
                    logger.info(f"Nomes das features carregados dos metadados: {len(feature_columns)} features")
                else:
                    # Tentar extrair nomes de features de outras fontes
                    feature_columns = extract_feature_names(client_id, X_train_client.shape[1])
            except:
                logger.warning("Não foi possível carregar nomes das features, usando nomes genéricos")
                feature_columns = [f"feature_{i}" for i in range(X_train_client.shape[1])]
            
            return X_train_client, False, None, y_train_client, feature_columns, X_train_client.copy()
        
        except Exception as e:
            logger.error(f"Erro ao carregar dados do cliente {client_id}: {str(e)}")
            logger.info("Usando dataset de exemplo pequeno")
            
            # Criar dataset de exemplo pequeno
            X_dummy = np.random.randn(10, 5).astype(np.float32)
            y_dummy = np.random.randint(0, 2, 10)
            dummy_features = [f"feature_{i}" for i in range(X_dummy.shape[1])]
            
            return X_dummy, False, None, y_dummy, dummy_features, X_dummy.copy()

def load_model(client_id):
    """Carrega o modelo treinado de um cliente específico"""
    # Obter diretório base do projeto
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, f"RLFE/client/results/client_{client_id}/model_client_{client_id}.pt")
    
    if not os.path.exists(model_path):
        # Tentar caminho relativo ao diretório atual
        model_path = f"client/results/client_{client_id}/model_client_{client_id}.pt"
        if not os.path.exists(model_path):
            logger.error(f"Modelo não encontrado em {model_path}")
            sys.exit(1)
        
    logger.info(f"Carregando modelo de {model_path}")
    
    try:
        # Primeiro, carregar os dados de treinamento para determinar a dimensão do modelo
        X_train, _, _, _, _, _ = load_train_data(client_id)
        input_size = X_train.shape[1]
        logger.info(f"Inicializando modelo com input_size={input_size} baseado nos dados de treinamento")
        
        # Importar a classe do modelo
        sys.path.append(os.path.join(os.path.dirname(__file__), '../client'))
        from utils import LinearRegressionModel
        
        # Criar um modelo com a arquitetura correta
        model = LinearRegressionModel(input_size=input_size)
        
        # Carregar os parâmetros do modelo
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        
        # Se for um OrderedDict de parâmetros Flower, converter para o formato esperado
        if isinstance(state_dict, dict) and all(key.startswith('0.') for key in state_dict.keys()):
            new_state_dict = {}
            for key, value in state_dict.items():
                new_key = key[2:]  # Remover o prefixo '0.'
                new_state_dict[new_key] = value
            logger.info("Convertendo formato de parâmetros para state_dict do PyTorch")
            state_dict = new_state_dict
        
        # Verificar as chaves do dicionário
        logger.info(f"Chaves do dicionário carregado: {list(state_dict.keys())}")
        
        # Verificar formatos dos parâmetros
        for key, param in state_dict.items():
            logger.info(f"Encontrado parâmetro tensor para chave {key} com formato {param.shape}")

        # Tentar carregar o modelo
        try:
            # Abordagem 1 - carregar manualmente
            model.linear.weight.data = state_dict['linear.weight']
            model.linear.bias.data = state_dict['linear.bias']
        except Exception as e:
            logger.warning(f"Erro na carga manual: {e}. Tentando outra abordagem.")
            # Abordagem 2 - usar o load_state_dict padrão
            model.load_state_dict(state_dict)
            logger.info("Modelo carregado usando load_state_dict padrão")
        
        # Definir o modo de avaliação
        model.eval()
        
        return model, X_train
        
    except Exception as e:
        logger.error(f"Erro ao carregar o modelo: {e}")
        sys.exit(1)

def select_instance(data, args, is_original_dataset=False):
    """
    Seleciona uma instância específica para explicação:
    - Se args.index for fornecido, usa esse índice
    - Se args.random for True, seleciona uma instância aleatória
    """
    if hasattr(args, 'index') and args.index is not None:
        # Usar o índice específico fornecido
        instance_idx = args.index
        if instance_idx >= len(data):
            logger.warning(f"Índice {instance_idx} fora dos limites. Usando índice 0 em vez disso.")
            instance_idx = 0
        logger.info(f"Selecionada instância com índice {instance_idx}")
    elif hasattr(args, 'random') and args.random:
        # Para verdadeira aleatoriedade, não devemos usar a seed fixa
        # Usamos uma combinação do tempo atual e da seed do usuário
        import time
        import random
        
        # Criar uma nova seed baseada no tempo atual e na seed do usuário
        current_time_ms = int(time.time() * 1000) % 10000
        random_seed = args.seed + current_time_ms if hasattr(args, 'seed') else current_time_ms
        
        # Configurar o gerador com a nova seed
        random.seed(random_seed)
        
        # Selecionar instância aleatória
        instance_idx = random.randint(0, len(data) - 1)
        
        # Armazenar o índice real para referência 
        args._instance_idx = instance_idx
        
        dataset_desc = "Dataset original completo" if is_original_dataset else "dados de treinamento do cliente"
        logger.info(f"Selecionada instância aleatória com índice {instance_idx} do {dataset_desc}")
    else:
        # Comportamento padrão
        instance_idx = 0
        logger.info("Nenhum método de seleção especificado, usando a primeira instância (índice 0)")
    
    return instance_idx
