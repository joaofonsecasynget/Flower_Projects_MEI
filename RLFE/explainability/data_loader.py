"""
Funções para carregamento de modelos e dados.
"""
import os
import sys
import logging
import numpy as np
import torch

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
        
        # Verificar se existe uma coluna de índice e preservá-la
        index_cols = [col for col in df.columns if col.lower() in ['índice', 'indice', 'index']]
        original_indices = None
        
        if index_cols:
            index_col = index_cols[0]
            logger.info(f"Preservando coluna de índice original: {index_col}")
            original_indices = df[index_col].values
            
        # Identificar a coluna target (assumindo que é a última coluna)
        target_col = df.columns[-1]  # Assume que a última coluna é o target
        logger.info(f"Coluna target identificada como '{target_col}'")
        
        # Preservar valores do target para comparação posterior
        target_values = df[target_col].values
        
        # Preservar nomes originais das colunas para referência futura
        original_columns = list(df.columns)
        
        # Remover colunas não-numéricas (exceto o target)
        non_numeric_cols = [col for col in df.columns if col != target_col and not pd.api.types.is_numeric_dtype(df[col])]
        
        if non_numeric_cols:
            logger.warning(f"Removendo {len(non_numeric_cols)} colunas não-numéricas: {', '.join(non_numeric_cols)}")
            df = df.drop(columns=non_numeric_cols)
        
        # Converter todas as colunas para numéricas
        for col in df.columns:
            if col != target_col and not pd.api.types.is_numeric_dtype(df[col]):
                logger.warning(f"Convertendo coluna '{col}' para numérica")
                df[col] = pd.to_numeric(df[col], errors='coerce')
                # Preencher valores NaN com 0
                df[col] = df[col].fillna(0)
        
        # Remover a coluna target
        df_X = df.drop(columns=[target_col])
        
        # Guardar os nomes das colunas
        feature_columns = list(df_X.columns)
        
        # Converter para um array NumPy
        X_original = df_X.values
        
        # Também carregamos os dados de treinamento do cliente para ter os mesmo formatos/features
        x_train_path = f"RLFE/client/reports/client_{client_id}/X_train.npy"
        if os.path.exists(x_train_path):
            X_train_client = np.load(x_train_path)
            logger.info(f"Dataset original tem {len(X_original)} instâncias, cliente {client_id} tem {len(X_train_client)} instâncias")
            
            # Verificar se os formatos são compatíveis
            if X_original.shape[1] != X_train_client.shape[1]:
                logger.warning(f"Incompatibilidade de features: dataset original tem {X_original.shape[1]} features, dados do cliente têm {X_train_client.shape[1]} features")
                
                # Se o dataset original tiver menos features, podemos adicionar colunas de zeros
                if X_original.shape[1] < X_train_client.shape[1]:
                    missing_features = X_train_client.shape[1] - X_original.shape[1]
                    logger.warning(f"Adicionando {missing_features} colunas de zeros ao dataset original para compatibilidade")
                    
                    # Adicionar colunas extras com zeros
                    padding = np.zeros((X_original.shape[0], missing_features), dtype=np.float32)
                    X_original = np.hstack((X_original, padding))
                    
                    # Adicionar nomes para as colunas extras
                    for i in range(missing_features):
                        feature_columns.append(f"extra_feature_{i}")
                    
                    logger.info(f"Dataset original agora tem {X_original.shape[1]} features, compatível com os dados do cliente")
        
        # Converter para float32 para garantir compatibilidade com o PyTorch
        X_original = X_original.astype(np.float32)
        
        # Retornar o dataset original, flag indicando uso do dataset original, índices originais, valores do target e nomes das colunas
        return X_original, True, original_indices, target_values, feature_columns
    else:
        # Comportamento original - carrega apenas os dados do cliente
        x_train_path = f"RLFE/client/reports/client_{client_id}/X_train.npy"
        if not os.path.exists(x_train_path):
            logger.error(f"Dados de treinamento não encontrados em {x_train_path}")
            sys.exit(1)
        
        if dataset_path:
            logger.warning(f"Dataset original não encontrado em {dataset_path}. Usando apenas dados do cliente.")
        else:
            logger.info(f"Carregando dados de treinamento do cliente de {x_train_path}")
            
        X_train = np.load(x_train_path)
        
        return X_train, False, None, None, None  # False indica que estamos usando apenas os dados do cliente

def load_model(client_id):
    """Carrega o modelo treinado de um cliente específico"""
    model_path = f"RLFE/client/results/client_{client_id}/model_client_{client_id}.pt"
    if not os.path.exists(model_path):
        logger.error(f"Modelo não encontrado em {model_path}")
        sys.exit(1)
        
    logger.info(f"Carregando modelo de {model_path}")
    
    try:
        # Primeiro, carregar os dados de treinamento para determinar a dimensão do modelo
        X_train, _, _, _, _ = load_train_data(client_id)
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
