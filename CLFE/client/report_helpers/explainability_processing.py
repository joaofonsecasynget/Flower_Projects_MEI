import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import torch
import time
import shap
import re
from datetime import datetime
import pandas as pd # Added pandas here as it's used within the functions
import random # Added random here as it's used within the functions
import sys

# Importar feature_metadata para denormalização
try:
    sys.path.append('/Users/joaofonseca/git/Flower_Projects_MEI/CLFE')
    from explainability.feature_metadata import feature_metadata
    FEATURE_METADATA_AVAILABLE = True
except ImportError:
    FEATURE_METADATA_AVAILABLE = False
    print("feature_metadata não pode ser importado, valores originais não estarão disponíveis")

from .formatting import extract_feature_name, format_value, get_interpretation

# Configuração do logger
logger = logging.getLogger(__name__) # Use __name__ for logger in modules

def generate_explainability(explainer, X_train, base_reports, sample_idx=None):
    """
    Gera explicabilidade LIME e SHAP para o modelo.
    
    Args:
        explainer: Instância de ModelExplainer
        X_train: Dados de treino ou caminho para um arquivo .npy contendo os dados
        base_reports: Diretório base para salvar as explicações
        sample_idx: Índice da amostra para explicar com LIME. Se None, seleciona aleatoriamente.
        
    Returns:
        tuple: Tupla com durações (lime_duration, shap_duration)
    """
    lime_duration = 0
    shap_duration = 0
    
    try:
        # Verificar se X_train é um caminho para arquivo e carregar os dados se necessário
        if isinstance(X_train, (str, Path)):
            logger.info(f"Carregando X_train do arquivo: {X_train}")
            X_train_path = Path(X_train)
            if X_train_path.exists():
                X_train = np.load(X_train_path)
                logger.info(f"Dados X_train carregados com sucesso: {X_train.shape}")
            else:
                logger.error(f"Arquivo X_train não encontrado em {X_train_path}")
                return lime_duration, shap_duration
        
        # Gerar LIME
        lime_start = time.time()
        try:
            # Garantir que o diretório base_reports existe
            os.makedirs(base_reports, exist_ok=True)
            
            # Escolher instância para explicar
            if sample_idx is None:
                # Selecionar instância aleatória
                # import random # No longer needed here, imported at module level
                random_seed = 42 + int(time.time() * 1000) % 10000
                random.seed(random_seed)
                sample_idx = random.randint(0, len(X_train) - 1)
                logger.info(f'Selecionada instância aleatória com índice {sample_idx}')
            else:
                logger.info(f'Usando instância com índice {sample_idx}')
            
            # Obter a instância para explicar
            instance = X_train[sample_idx].reshape(1, -1)[0]
            
            # Obter previsão do modelo para esta instância
            instance_tensor = torch.tensor(instance, dtype=torch.float32).to(explainer.device)
            with torch.no_grad():
                prediction = explainer.model(instance_tensor.unsqueeze(0)).item()
            
            # Tentar carregar o dataset original para obter valores não normalizados
            try:
                # Tentar múltiplos caminhos possíveis para o dataset original
                possible_paths = [
                    '/app/DatasetIOT/transformed_dataset_imeisv_8642840401612300.csv',  # Caminho no Docker
                    '../DatasetIOT/transformed_dataset_imeisv_8642840401612300.csv',      # Caminho relativo
                    '/Users/joaofonseca/git/Flower_Projects_MEI/DatasetIOT/transformed_dataset_imeisv_8642840401612300.csv'  # Caminho absoluto local
                ]
                
                original_dataset_path = None
                for path in possible_paths:
                    if os.path.exists(path):
                        original_dataset_path = path
                        logger.info(f"Dataset original encontrado em: {path}")
                        break
                
                if original_dataset_path:
                    logger.info(f"Carregando valores originais de {original_dataset_path}")
                    # import pandas as pd # No longer needed here, imported at module level
                    df_original = pd.read_csv(original_dataset_path)
                    
                    # Extrair features temporais do timestamp
                    if '_time' in df_original.columns:
                        logger.info(f"Extraindo features temporais de _time para valores originais")
                        # Converter para datetime
                        try:
                            df_original['_time'] = pd.to_datetime(df_original['_time'])
                        except:
                            try:
                                logger.info("Standard datetime parsing failed, trying with ISO8601 format")
                                df_original['_time'] = pd.to_datetime(df_original['_time'], format='ISO8601')
                            except Exception as e:
                                logger.error(f"Failed to parse _time column as datetime: {e}")
                        
                        # Extrair features temporais
                        if pd.api.types.is_datetime64_any_dtype(df_original['_time']):
                            df_original['hour'] = df_original['_time'].dt.hour
                            df_original['dayofweek'] = df_original['_time'].dt.dayofweek
                            df_original['day'] = df_original['_time'].dt.day
                            df_original['month'] = df_original['_time'].dt.month
                            df_original['is_weekend'] = (df_original['dayofweek'] >= 5).astype(int)
                            logger.info("Features temporais extraídas com sucesso do dataset original")
                            
                            # Gerar instance_info.json com todas as features após criação das features temporais
                            try:
                                # Obter a instância original com as features temporais incluídas
                                instance_row = df_original.iloc[sample_idx]
                                
                                # Colunas a serem excluídas na exportação
                                exclude_cols = ['indice', 'imeisv', 'index', '_time']
                                
                                # Criar dicionário com todos os valores de features
                                feature_values = {}
                                
                                # Para cada feature disponível no dataset
                                for col in df_original.columns:
                                    if col not in exclude_cols:
                                        value = instance_row[col]
                                        if isinstance(value, (np.integer, np.floating)):
                                            value = float(value)
                                        else:
                                            value = str(value)
                                        
                                        feature_values[col] = {
                                            'original': value
                                        }
                                
                                # Adicionar valores normalizados quando disponíveis
                                for i, name in enumerate(explainer.feature_names):
                                    norm_val = instance[i]
                                    if isinstance(norm_val, (np.integer, np.floating)):
                                        norm_val = float(norm_val)
                                        
                                    if name in feature_values:
                                        feature_values[name]['normalized'] = round(norm_val, 5)
                                    else:
                                        feature_values[name] = {
                                            'normalized': round(norm_val, 5)
                                        }
                                
                                # Criar estrutura completa do arquivo
                                instance_info = {
                                    'original_index': sample_idx,
                                    'prediction': round(float(prediction), 5),
                                    'predicted_class': 'Ataque' if prediction > 0.5 else 'Normal',
                                    'prediction_confidence': round((prediction if prediction > 0.5 else 1 - prediction) * 100, 2),
                                    'original_target': int(instance_row['attack']) if 'attack' in instance_row else None,
                                    'original_target_class': 'Ataque' if 'attack' in instance_row and instance_row['attack'] == 1 else 'Normal',
                                    'prediction_correct': (prediction > 0.5) == (instance_row['attack'] == 1) if 'attack' in instance_row else None,
                                    'feature_count': len(feature_values),
                                    'features': feature_values
                                }
                                
                                # Salvar o arquivo
                                instance_info_path = base_reports / 'instance_info.json'
                                with open(instance_info_path, 'w') as f:
                                    json.dump(instance_info, f, indent=4, default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else bool(o) if isinstance(o, np.bool_) else repr(o))
                                logger.info(f"Arquivo instance_info.json com todos os valores de features criado em {instance_info_path}")
                            except Exception as e:
                                logger.error(f"Erro ao gerar instance_info.json: {e}", exc_info=True)
                    
                    # Se tivermos o índice original, usamos para buscar os valores originais
                    original_instance = df_original.iloc[sample_idx]
                    
                    # Obter valor target original se disponível
                    original_target = None
                    original_target_class = None
                    prediction_correct = None
                    
                    if 'attack' in df_original.columns:
                        original_target = int(original_instance['attack'])
                        original_target_class = "Ataque" if original_target == 1 else "Normal"
                        # Determinar se a previsão está correta
                        # Aqui, estamos assumindo que a previsão é a probabilidade da classe positiva (Ataque)
                        # A conversão para classe predita (0 ou 1) dependerá de um limiar (threshold), tipicamente 0.5
                        predicted_class_label = 1 if prediction > 0.5 else 0 
                        prediction_correct = (predicted_class_label == original_target)
                    else:
                        logger.warning("Coluna 'attack' não encontrada no dataset original para determinar o target.")
                    
                    # Criar um dicionário com os valores originais das features usadas no modelo
                    original_values_dict = {}
                    for feature_name in explainer.feature_names:
                        if feature_name in original_instance:
                            original_values_dict[feature_name] = original_instance[feature_name]
                        elif feature_name in df_original.columns: # Caso de features como 'hour' que foram criadas
                            original_values_dict[feature_name] = df_original[feature_name].iloc[sample_idx]
                        else:
                            original_values_dict[feature_name] = "N/A (original)"
                else:
                    logger.warning(f"Dataset original não encontrado em {original_dataset_path}, não será possível mostrar valores originais.")
                    original_values_dict = None
                    original_target = None
                    original_target_class = None
                    prediction_correct = None

            except Exception as e:
                logger.error(f"Erro ao carregar ou processar o dataset original: {e}", exc_info=True)
                original_values_dict = None
                original_target = None
                original_target_class = None
                prediction_correct = None

            lime_exp = explainer.explain_lime(instance, num_features=len(explainer.feature_names))
            
            # Adicionar informações ao LIME explainer (se não existirem)
            lime_exp.payload = getattr(lime_exp, 'payload', {})
            lime_exp.payload['original_values'] = original_values_dict
            lime_exp.payload['original_target'] = original_target
            lime_exp.payload['original_target_class'] = original_target_class
            lime_exp.payload['prediction_correct'] = prediction_correct
            lime_exp.payload['sample_index_explained'] = sample_idx
            lime_exp.payload['predicted_probability'] = prediction
            
            # Salvar LIME explainer completo com payload
            lime_explainer_path = base_reports / 'lime_explainer_with_payload.json'
            try:
                explanation_data_to_save = {
                    'explanation': lime_exp.as_list(),
                    'payload': lime_exp.payload
                }
                with open(lime_explainer_path, 'w') as f:
                    json.dump(explanation_data_to_save, f, indent=4, default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else bool(o) if isinstance(o, np.bool_) else repr(o))
                logger.info(f"LIME explainer com payload salvo em {lime_explainer_path}")
                
                # NOVO: Salvar explained_instance_info.json para o relatório HTML
                instance_info_path = base_reports / "explained_instance_info.json"
                try:
                    # Nova abordagem: manter valores originais E normalizados
                    
                    # Dicionário para armazenar ambos os conjuntos de valores
                    features_dict = {}
                    original_values = {}
                    normalized_values = {}
                    
                    # Primeiro, carregar todos os valores normalizados
                    for i, name in enumerate(explainer.feature_names):
                        norm_val = instance[i]
                        if isinstance(norm_val, (np.integer, np.floating)):
                            norm_val = float(norm_val)
                        
                        # Apenas incluir se o valor for diferente de zero
                        if abs(norm_val) > 0.001:  # Filtrar valores próximos de zero
                            normalized_values[name] = round(norm_val, 5)
                    
                    # Depois, carregar os valores originais quando disponíveis
                    if original_values_dict:
                        # Remover colunas que não são features
                        cols_to_drop = []
                        for col in ['indice', 'imeisv', 'index', 'attack', '_time']:
                            if col in original_instance.index:
                                cols_to_drop.append(col)
                        
                        # Criar Series apenas com as features
                        original_features = original_instance.drop(cols_to_drop) if cols_to_drop else original_instance
                        
                        # Carregar valores originais disponíveis
                        for name in explainer.feature_names:
                            if name in original_features:
                                val = original_features[name]
                                if isinstance(val, (np.integer, np.floating)):
                                    val = float(val)
                                
                                # Apenas incluir se o valor for diferente de zero
                                if abs(val) > 0.001:
                                    original_values[name] = round(val, 5)
                        
                        logger.info(f"Carregados {len(original_values)} valores originais e {len(normalized_values)} valores normalizados")
                    else:
                        logger.warning(f"Dataset original não encontrado, apenas valores normalizados disponíveis")
                    
                    # Combinar em um único dicionário estruturado com 'original' e 'normalized'
                    for name in set(list(original_values.keys()) + list(normalized_values.keys())):
                        features_dict[name] = {}
                        if name in normalized_values:
                            features_dict[name]['normalized'] = normalized_values[name]
                        if name in original_values:
                            features_dict[name]['original'] = original_values[name]
                        # Se só tiver o valor normalizado, copiar para 'original' também como fallback
                        elif name in normalized_values:
                            features_dict[name]['original'] = normalized_values[name]
                    
                    logger.info(f"Total de {len(features_dict)} features incluídas no explained_instance_info.json")
                    
                    # Determinar a classe predita
                    predicted_class = "Ataque" if prediction > 0.5 else "Normal"
                    prediction_confidence = round((prediction if prediction > 0.5 else 1 - prediction) * 100, 2)
                    
                    # Criar dicionário completo de informações da instância
                    instance_info = {
                        'original_index': sample_idx,
                        'prediction': round(float(prediction), 5),
                        'predicted_class': predicted_class,
                        'prediction_confidence': prediction_confidence,
                        'original_target': original_target,
                        'original_target_class': original_target_class,
                        'prediction_correct': prediction_correct,
                        'feature_count': len(explainer.feature_names),
                        'features': features_dict
                    }
                    
                    # Salvar as informações
                    try:
                        with open(instance_info_path, 'w') as f:
                            json.dump(instance_info, f, indent=4, default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else bool(o) if isinstance(o, np.bool_) else repr(o))
                        
                        logger.info(f"Informações da instância salvas com sucesso em {instance_info_path}")
                        
                        # Verificar se o arquivo foi realmente criado
                        if os.path.exists(instance_info_path):
                            logger.info(f"Arquivo {instance_info_path} existe e tem tamanho {os.path.getsize(instance_info_path)} bytes")
                        else:
                            logger.error(f"Arquivo {instance_info_path} não existe após a tentativa de criação")
                            
                            # Tentar caminho alternativo como fallback
                            fallback_path = os.path.join(str(base_reports), "explained_instance_info.json")
                            logger.info(f"Tentando salvar em caminho alternativo: {fallback_path}")
                            with open(fallback_path, 'w') as f:
                                json.dump(instance_info, f, indent=4, default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else bool(o) if isinstance(o, np.bool_) else repr(o))
                            logger.info(f"Informações salvas em caminho alternativo: {fallback_path}")
                    except Exception as e:
                        logger.error(f"Erro ao escrever no arquivo {instance_info_path}: {e}", exc_info=True)
                except Exception as e:
                    logger.error(f"Erro ao salvar explained_instance_info.json: {e}", exc_info=True)
                
                # Código de fallback removido para simplificar
            except Exception as e:
                logger.error(f"Erro ao salvar LIME explainer com payload: {e}", exc_info=True)

            # Salvar visualização LIME
            lime_png_path = base_reports / 'lime_final.png'
            
            # Não gerar o arquivo HTML indesejado
            # lime_exp.save_to_file(str(base_reports / 'lime_final.html'))
            
            # Usar nossa visualização personalizada
            explainer.create_custom_lime_visualization(lime_exp, lime_png_path)
            
            logger.info(f'LIME explanation generated and saved to {lime_png_path}')
        except Exception as e:
            logger.error(f'Error generating LIME explanation: {e}', exc_info=True)
        finally:
            lime_duration = time.time() - lime_start
        
        # Gerar SHAP
        shap_start = time.time()
        try:
            # Executar SHAP
            shap_values = explainer.explain_shap(X_train)
            
            # Salvar valores SHAP como arquivo .npy
            shap_values_path = base_reports / 'shap_values.npy'
            np.save(shap_values_path, shap_values)
            
            # Gerar visualização SHAP
            shap_png_path = base_reports / 'shap_final.png'
            
            try:
                # Verificar se o método personalizado existe
                if hasattr(explainer, 'create_custom_shap_visualization'):
                    explainer.create_custom_shap_visualization(shap_values, shap_png_path)
                else:
                    # Criar visualização SHAP manualmente se o método não existir
                    plt.figure(figsize=(12, 8))
                    
                    # Calcular importância média das features
                    if isinstance(shap_values, list):
                        feature_importance = np.abs(shap_values[1]).mean(0)
                    else:
                        feature_importance = np.abs(shap_values).mean(0)
                    
                    # Verificar dimensionalidade
                    if feature_importance.ndim > 1:
                        feature_importance = feature_importance.mean(axis=1)
                    
                    # Ordenar features por importância
                    indices = np.argsort(feature_importance)
                    n_features = min(20, len(feature_importance))  # Limitar a 20 features
                    plt.barh(range(n_features), feature_importance[indices[-n_features:]], color='#1E88E5')
                    plt.yticks(range(n_features), [explainer.feature_names[i] for i in indices[-n_features:]])
                    plt.xlabel('Importância Média (|SHAP|)')
                    plt.title('Importância das Features (SHAP)')
                    plt.tight_layout()
                    plt.savefig(shap_png_path, dpi=150, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                logger.error(f"Erro ao criar visualização SHAP: {e}")
                # Criar um gráfico simples como fallback
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, "Não foi possível gerar visualização SHAP", 
                        horizontalalignment='center', verticalalignment='center', fontsize=14)
                plt.savefig(shap_png_path)
                plt.close()
            
            logger.info(f'SHAP explanation generated and saved to {shap_png_path}')
        except Exception as e:
            logger.error(f'Error generating SHAP explanation: {e}', exc_info=True)
        finally:
            shap_duration = time.time() - shap_start
        
        return lime_duration, shap_duration
    except Exception as e:
        logger.error(f'Error in generate_explainability: {e}', exc_info=True)
        return 0, 0

def generate_explainability_formatado(explainer, X_train, base_reports, sample_idx=None):
    """
    Gera explicabilidade LIME formatada para o modelo.
    
    Args:
        explainer: Instância de ModelExplainer
        X_train: Dados de treino ou caminho para um arquivo .npy contendo os dados
        base_reports: Diretório base para salvar as explicações
        sample_idx: Índice da amostra para explicar com LIME. Se None, seleciona aleatoriamente.
        
    Returns:
        tuple: Tupla com durações (lime_duration, shap_duration)
    """
    lime_duration = 0
    shap_duration = 0 # Embora esta função foque em LIME, mantemos a estrutura para consistência
    
    try:
        # Verificar se X_train é um caminho para arquivo e carregar os dados se necessário
        if isinstance(X_train, (str, Path)):
            logger.info(f"Carregando X_train do arquivo: {X_train}")
            X_train_path = Path(X_train)
            if X_train_path.exists():
                X_train = np.load(X_train_path)
                logger.info(f"Dados X_train carregados com sucesso: {X_train.shape}")
            else:
                logger.error(f"Arquivo X_train não encontrado em {X_train_path}")
                return lime_duration, shap_duration
        
        # Gerar LIME
        lime_start = time.time()
        try:
            # Escolher instância para explicar
            if sample_idx is None:
                # import random # No longer needed here, imported at module level
                random_seed = 42 + int(time.time() * 1000) % 10000 # Usar semente diferente ou coordenada
                random.seed(random_seed)
                sample_idx = random.randint(0, len(X_train) - 1)
                logger.info(f'Selecionada instância aleatória com índice {sample_idx} para LIME formatado')
            else:
                logger.info(f'Usando instância com índice {sample_idx} para LIME formatado')
            
            instance = X_train[sample_idx].reshape(1, -1)[0]
            instance_tensor = torch.tensor(instance, dtype=torch.float32).to(explainer.device)
            with torch.no_grad():
                prediction = explainer.model(instance_tensor.unsqueeze(0)).item()

            # Tentar carregar o dataset original para obter valores não normalizados
            original_values_dict = None
            original_target = None
            original_target_class = None
            prediction_correct = None
            instance_source_info = "Dados de Treino do Cliente (normalizados)"

            try:
                original_dataset_path = '/app/DatasetIOT/transformed_dataset_imeisv_8642840401612300.csv'
                if os.path.exists(original_dataset_path):
                    logger.info(f"Carregando valores originais de {original_dataset_path} para LIME formatado")
                    # import pandas as pd # No longer needed here, imported at module level
                    df_original = pd.read_csv(original_dataset_path)
                    instance_source_info = f"Dataset Original ({original_dataset_path}), índice {sample_idx}"

                    # Extrair features temporais do timestamp
                    if '_time' in df_original.columns:
                        logger.info(f"Extraindo features temporais de _time para valores originais (LIME formatado)")
                        try:
                            df_original['_time'] = pd.to_datetime(df_original['_time'])
                        except:
                            try:
                                logger.info("Standard datetime parsing failed, trying with ISO8601 format (LIME formatado)")
                                df_original['_time'] = pd.to_datetime(df_original['_time'], format='ISO8601')
                            except Exception as e:
                                logger.error(f"Failed to parse _time column as datetime (LIME formatado): {e}")
                        
                        if pd.api.types.is_datetime64_any_dtype(df_original['_time']):
                            df_original['hour'] = df_original['_time'].dt.hour
                            df_original['dayofweek'] = df_original['_time'].dt.dayofweek
                            df_original['day'] = df_original['_time'].dt.day
                            df_original['month'] = df_original['_time'].dt.month
                            df_original['is_weekend'] = (df_original['dayofweek'] >= 5).astype(int)
                            logger.info("Features temporais extraídas com sucesso do dataset original (LIME formatado)")

                    original_instance = df_original.iloc[sample_idx]
                    if 'attack' in df_original.columns:
                        original_target = int(original_instance['attack'])
                        original_target_class = "Ataque" if original_target == 1 else "Normal"
                        predicted_class_label = 1 if prediction > 0.5 else 0 
                        prediction_correct = (predicted_class_label == original_target)
                    else:
                        logger.warning("Coluna 'attack' não encontrada no dataset original para determinar o target (LIME formatado).")
                    
                    original_values_dict = {}
                    for feature_name in explainer.feature_names:
                        if feature_name in original_instance:
                            original_values_dict[feature_name] = original_instance[feature_name]
                        elif feature_name in df_original.columns:
                            original_values_dict[feature_name] = df_original[feature_name].iloc[sample_idx]
                        else:
                            original_values_dict[feature_name] = "N/A (original)"
                else:
                    logger.warning(f"Dataset original não encontrado em {original_dataset_path}, não será possível mostrar valores originais (LIME formatado).")
            except Exception as e:
                logger.error(f"Erro ao carregar ou processar o dataset original (LIME formatado): {e}", exc_info=True)

            lime_exp = explainer.explain_lime(instance, num_features=len(explainer.feature_names))
            
            # Formatar e salvar LIME como texto
            lime_txt_path = base_reports / 'lime_explanation_formatado.txt'
            try:
                with open(lime_txt_path, 'w', encoding='utf-8') as f:
                    f.write(f"Explicação LIME para a instância índice: {sample_idx}\n")
                    f.write(f"Fonte da Instância: {instance_source_info}\n")
                    f.write(f"Probabilidade Predita (Classe Ataque): {prediction:.4f}\n")
                    if original_target_class is not None:
                        f.write(f"Classe Original: {original_target_class} (Valor: {original_target})\n")
                        if prediction_correct is not None:
                            f.write(f"Previsão Correta: {'Sim' if prediction_correct else 'Não'}\n")
                    f.write("\nContribuições das Features (LIME):\n")
                    f.write("-------------------------------------\n")
                    f.write(f"Feature{'':<23} | Valor Normalizado | Valor Original{'':<8} | Impacto (LIME){'':<5} | Interpretação\n")
                    f.write("----------------------------------------------------------------------------------------------------------\n")

                    # Agrupar por categorias para garantir a ordem e inclusão de todas
                    feature_categories = {
                        # Categorias conforme definido anteriormente
                        'dl_bitrate': [], 'ul_bitrate': [], 'cell_x_dl_retx': [], 
                        'cell_x_dl_tx': [], 'cell_x_ul_retx': [], 'cell_x_ul_tx': [],
                        'ul_total_bytes_non_incr': [], 'dl_total_bytes_non_incr': [],
                        'time_features': [], 'other': [] # fallback
                    }
                    
                    # Mapeamento de prefixos para categorias
                    category_map = {
                        'dl_br_': 'dl_bitrate',
                        'ul_br_': 'ul_bitrate',
                        'dl_rx_': 'cell_x_dl_retx',
                        'dl_tx_': 'cell_x_dl_tx',
                        'ul_rx_': 'cell_x_ul_retx',
                        'ul_tx_': 'cell_x_ul_tx',
                        'ul_b_': 'ul_total_bytes_non_incr', # Ajuste conforme o nome real
                        'dl_b_': 'dl_total_bytes_non_incr', # Ajuste conforme o nome real
                    }
                    time_feature_names = ['hour', 'dayofweek', 'day', 'month', 'is_weekend']

                    temp_explanation_list = [] # Para ordenar por relevância dentro das categorias depois
                    for feature_idx, impact in lime_exp.as_list():
                        raw_feature_name = explainer.feature_names[feature_idx]
                        processed_feature_name = extract_feature_name(raw_feature_name) 
                        normalized_value = instance[feature_idx]
                        original_value_display = "N/A"
                        if original_values_dict and processed_feature_name in original_values_dict:
                            original_value_display = format_value(original_values_dict[processed_feature_name])
                        elif original_values_dict and raw_feature_name in original_values_dict: # Tentar com nome raw se não achou com processado
                             original_value_display = format_value(original_values_dict[raw_feature_name])

                        interpretation = get_interpretation(impact)
                        temp_explanation_list.append({
                            'raw_name': raw_feature_name,
                            'processed_name': processed_feature_name,
                            'norm_val': format_value(normalized_value),
                            'orig_val': original_value_display,
                            'impact': impact, # Manter valor numérico para ordenação
                            'interpretation': interpretation
                        })
                    
                    # Ordenar a lista completa por magnitude do impacto (absoluto e decrescente)
                    sorted_explanation_list = sorted(temp_explanation_list, key=lambda x: abs(x['impact']), reverse=True)

                    # Atribuir a categorias e manter a ordem de relevância global
                    categorized_features_output = []
                    for item in sorted_explanation_list:
                        category = 'other' # Default category
                        if item['processed_name'] in time_feature_names:
                            category = 'time_features'
                        else:
                            for prefix, cat_name in category_map.items():
                                if item['processed_name'].startswith(prefix):
                                    category = cat_name
                                    break
                        
                        # Construir a linha formatada
                        line = f"{item['processed_name']:<30} | {item['norm_val']:<15} | {item['orig_val']:<20} | {format_value(item['impact']):<20} | {item['interpretation']}"
                        categorized_features_output.append(line)
                    
                    # Escrever no arquivo
                    for line in categorized_features_output:
                        f.write(line + "\n")

                logger.info(f'LIME explanation (formatted) saved to {lime_txt_path}')
            except Exception as e:
                logger.error(f'Erro ao criar versão formatada da explicação LIME: {e}', exc_info=True)

        except Exception as e:
            logger.error(f'Error generating LIME explanation for formatted output: {e}', exc_info=True)
        finally:
            lime_duration = time.time() - lime_start
        
        # Esta função é focada no LIME formatado, SHAP não é regerado aqui para evitar redundância
        # Se SHAP fosse necessário especificamente para este fluxo, seria adicionado aqui.
        # shap_duration = ... 
        
        return lime_duration, shap_duration # Retorna duração LIME e 0 para SHAP
    except Exception as e:
        logger.error(f'Error in generate_explainability_formatado: {e}', exc_info=True)
        return 0, 0
