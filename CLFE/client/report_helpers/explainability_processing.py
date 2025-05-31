# DEBUG-CACHE-INVALIDATE-V7

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
import pandas as pd
import random
import sys

# Importar feature_metadata para denormalização
try:
    # Tentar um import relativo primeiro, assumindo uma estrutura de pacotes
    # Se explainability_processing.py está em CLFE/client/report_helpers/
    # e feature_metadata.py está em CLFE/explainability/
    # O caminho relativo de report_helpers para CLFE é .. (duas vezes para cima)
    # e depois para explainability.
    # No entanto, o import original usa sys.path.append com um caminho absoluto.
    # Vou manter o sys.path.append para consistência com o código original,
    # mas comentando uma alternativa mais robusta.
    # current_dir = Path(__file__).parent # CLFE/client/report_helpers
    # project_root = current_dir.parents[2] # CLFE
    # sys.path.append(str(project_root))
    sys.path.append('/Users/joaofonseca/git/Flower_Projects_MEI/CLFE') # Mantendo como estava
    from explainability.feature_metadata import feature_metadata
    FEATURE_METADATA_AVAILABLE = True
    # logger.info("feature_metadata importado com sucesso via sys.path absoluto.") # Logger não está definido ainda
except ImportError:
    FEATURE_METADATA_AVAILABLE = False
    print("feature_metadata não pode ser importado. Valores originais podem não estar disponíveis ou ser limitados.")

from .formatting import extract_feature_name, format_value, get_interpretation

# Configuração do logger
logger = logging.getLogger(__name__)

if FEATURE_METADATA_AVAILABLE:
    logger.info("feature_metadata importado com sucesso e está disponível.")
else:
    logger.warning("feature_metadata não pôde ser importado. Funcionalidades de desnormalização podem ser limitadas.")

def generate_explainability(
    explainer,
    X_background_path_or_data,
    base_reports,
    instance_to_explain_node=None,
    scaler=None,
    feature_names=None,
    original_target_value_for_explained_instance=None,
    predicted_value_for_explained_instance=None, # Raw prediction output (float)
    original_target_class_str_for_explained_instance=None
):
    """
    Gera explicabilidade LIME (formatada em texto) e SHAP (plots e valores) para o modelo.

    Args:
        explainer: Instância de ModelExplainer.
        X_background_path_or_data: Dados de treino/fundo ou caminho para arquivo .npy.
        base_reports: Diretório base para salvar as explicações.
        instance_to_explain_node: Instância específica (normalizada) a ser explicada.
        scaler: Scaler treinado para desnormalizar a instância.
        feature_names: Lista de nomes de features correspondentes à instância.
        original_target_value_for_explained_instance (int, optional): Valor real do target para a instância explicada.
        predicted_value_for_explained_instance (float, optional): Saída de previsão raw do modelo para a instância.
        original_target_class_str_for_explained_instance (str, optional): Classe original da instância como string.

    Returns:
        tuple: Tupla com durações (lime_duration, shap_duration).
    """
    lime_duration = 0
    shap_duration = 0

    try:
        X_background_data = None
        if isinstance(X_background_path_or_data, (str, Path)):
            logger.info(f"Carregando dados de fundo do arquivo: {X_background_path_or_data}")
            X_background_path = Path(X_background_path_or_data)
            if X_background_path.exists():
                X_background_data = np.load(X_background_path)
                logger.info(f"Dados de fundo carregados com sucesso: {X_background_data.shape}")
            else:
                logger.error(f"Arquivo de dados de fundo não encontrado em {X_background_path}")
                return lime_duration, shap_duration
        elif isinstance(X_background_path_or_data, np.ndarray):
            X_background_data = X_background_path_or_data
            logger.info(f"Usando dados de fundo fornecidos diretamente (ndarray): {X_background_data.shape}")
        else:
            logger.error("X_background_path_or_data deve ser um caminho de arquivo ou numpy array.")
            return lime_duration, shap_duration

        if X_background_data is None or X_background_data.size == 0:
            logger.error("Dados de fundo não disponíveis ou vazios.")
            return lime_duration, shap_duration

        target_instance_normalized = None
        if instance_to_explain_node is not None:
            target_instance_normalized = instance_to_explain_node
            logger.info("Usando instância específica fornecida (normalizada) para explicações.")
        elif X_background_data.shape[0] > 0:
            random_idx_bg = random.randint(0, X_background_data.shape[0] - 1)
            target_instance_normalized = X_background_data[random_idx_bg]
            logger.info(f"Nenhuma instância específica fornecida. Selecionando aleatoriamente amostra {random_idx_bg} dos dados de fundo.")
        else:
            logger.error("Dados de fundo vazios e nenhuma instância específica para explicar.")
            return lime_duration, shap_duration

        if target_instance_normalized.ndim > 1:
            target_instance_normalized = target_instance_normalized.flatten()

        os.makedirs(base_reports, exist_ok=True)

        original_target_info_dict = None
        if original_target_value_for_explained_instance is not None and \
           predicted_value_for_explained_instance is not None and \
           original_target_class_str_for_explained_instance is not None:

            predicted_class = 1 if predicted_value_for_explained_instance > 0.5 else 0
            prediction_is_correct = (predicted_class == int(original_target_value_for_explained_instance))

            original_target_info_dict = {
                'original_target_class': original_target_class_str_for_explained_instance,
                'original_target': int(original_target_value_for_explained_instance),
                'prediction_correct': prediction_is_correct,
                'predicted_raw_value': predicted_value_for_explained_instance # Adicionado para passar para LIME formatado
            }
            logger.info(f"Informação do target original para LIME formatado: {original_target_info_dict}")
        else:
            missing_oti = []
            if original_target_value_for_explained_instance is None: missing_oti.append("original_target_value")
            if predicted_value_for_explained_instance is None: missing_oti.append("predicted_value")
            if original_target_class_str_for_explained_instance is None: missing_oti.append("original_target_class_str")
            logger.warning(f"Informação do target original incompleta ({', '.join(missing_oti)}). Não será totalmente incluída na explicação LIME formatada.")

        lime_duration_call, _ = generate_explainability_formatado(
            explainer=explainer,
            X_background_path_or_data=X_background_data,
            base_reports=base_reports,
            instance_to_explain_node=target_instance_normalized,
            scaler=scaler,
            feature_names=feature_names,
            original_target_info=original_target_info_dict
        )
        lime_duration = lime_duration_call

        shap_start = time.time()
        try:
            if X_background_data.shape[0] > 100:
                shap_background = shap.sample(X_background_data, 100)
                logger.info(f"SHAP usará uma amostra de 100 instâncias do background data ({X_background_data.shape[0]} disponíveis).")
            else:
                shap_background = X_background_data
                logger.info(f"SHAP usará todo o background data disponível ({X_background_data.shape[0]} instâncias).")

            shap_explainer_obj = explainer.explain_shap(shap_background)
            instance_for_shap = target_instance_normalized.reshape(1, -1)
            shap_values_instance = shap_explainer_obj(instance_for_shap)

            shap_values_path = base_reports / 'shap_values.npy'
            np.save(shap_values_path, shap_values_instance.values)
            logger.info(f'SHAP values saved to {shap_values_path}')
            
            final_feature_names_for_shap = None
            num_features_instance = instance_for_shap.shape[1]
            if feature_names is not None and len(feature_names) == num_features_instance:
                final_feature_names_for_shap = feature_names
            elif hasattr(explainer, 'feature_names') and explainer.feature_names is not None and len(explainer.feature_names) == num_features_instance:
                final_feature_names_for_shap = explainer.feature_names
            else:
                final_feature_names_for_shap = [f"feature_{i}" for i in range(num_features_instance)]
                logger.warning(f"Nomes de features para SHAP não correspondem ou não disponíveis. Usando genéricos.")

            plt.clf() # Limpar figura atual
            shap.summary_plot(shap_values_instance.values, instance_for_shap, feature_names=final_feature_names_for_shap, show=False)
            plt.title('SHAP Summary Plot (Instância)')
            plt.tight_layout()
            shap_summary_path = base_reports / 'shap_summary_plot_instance.png' # Nome alterado para diferenciar do global
            plt.savefig(shap_summary_path)
            plt.close() # Fechar a figura
            logger.info(f'SHAP summary plot (instância) saved to {shap_summary_path}')

            plt.clf() # Limpar figura atual
            waterfall_explanation = shap.Explanation(
                values=shap_values_instance.values[0],
                base_values=shap_values_instance.base_values[0], # Assegurar que é um escalar
                data=shap_values_instance.data[0],
                feature_names=final_feature_names_for_shap
            )
            shap.plots.waterfall(waterfall_explanation, max_display=20, show=False)
            plt.title('SHAP Waterfall Plot (Instância)')
            plt.tight_layout()
            shap_waterfall_path = base_reports / 'shap_waterfall_plot.png'
            plt.savefig(shap_waterfall_path)
            plt.close() # Fechar a figura
            logger.info(f'SHAP waterfall plot saved to {shap_waterfall_path}')

        except Exception as e:
            logger.error(f'Error generating SHAP explanation: {e}', exc_info=True)
        finally:
            shap_duration = time.time() - shap_start

    except Exception as e:
        logger.error(f'Error in generate_explainability: {e}', exc_info=True)
        return 0, 0

    return lime_duration, shap_duration


def generate_explainability_formatado(explainer, X_background_path_or_data, base_reports, instance_to_explain_node=None, scaler=None, feature_names=None, original_target_info=None):
    """
    Gera uma versão formatada da explicação LIME, focada em clareza e usabilidade.
    Salva a explicação em formato de texto.

    Args:
        explainer: Instância de ModelExplainer.
        X_background_path_or_data: Dados de treino/fundo ou caminho para arquivo .npy.
        base_reports: Diretório base para salvar as explicações.
        instance_to_explain_node: Instância específica (normalizada) a ser explicada.
        scaler: Scaler treinado para desnormalizar a instância.
        feature_names: Lista de nomes de features correspondentes à instância (argumento fornecido).
        original_target_info: Dicionário com {'original_target_class': str, 
                                           'original_target': int, 
                                           'prediction_correct': bool,
                                           'predicted_raw_value': float}.
                              Se None, estas informações não serão incluídas.

    Returns:
        tuple: Duração LIME e SHAP (SHAP é 0 pois não é gerado aqui).
    """
    lime_duration = 0
    shap_duration = 0
    lime_start = time.time()

    client_id_log = getattr(explainer, 'cid', 'UNKNOWN_LIME_FORMAT')
    
    logger.info(f"LIME_FORMATADO [{client_id_log}]: Iniciando formatação. Base reports: {base_reports}")

    if isinstance(X_background_path_or_data, np.ndarray):
        logger.info(f"LIME_FORMATADO [{client_id_log}]: X_background_data (recebido como X_background_path_or_data) é um ndarray com forma: {X_background_path_or_data.shape}")
    elif isinstance(X_background_path_or_data, (str, Path)):
        logger.info(f"LIME_FORMATADO [{client_id_log}]: X_background_path_or_data é um caminho: {X_background_path_or_data}")
    else:
        logger.warning(f"LIME_FORMATADO [{client_id_log}]: Tipo de X_background_path_or_data não reconhecido: {type(X_background_path_or_data)}")

    if instance_to_explain_node is not None:
        if isinstance(instance_to_explain_node, np.ndarray):
            logger.info(f"LIME_FORMATADO [{client_id_log}]: instance_to_explain_node (instância única para formatação) é um ndarray com forma: {instance_to_explain_node.shape}")
        else:
            logger.info(f"LIME_FORMATADO [{client_id_log}]: instance_to_explain_node (instância única para formatação) tem tipo: {type(instance_to_explain_node)}")
    else:
        logger.info(f"LIME_FORMATADO [{client_id_log}]: instance_to_explain_node não foi fornecido. Uma instância aleatória dos dados de fundo será usada (lógica existente). ")

    try:
        X_background_data = None
        if isinstance(X_background_path_or_data, (str, Path)):
            logger.info(f"Lendo dados de fundo de: {X_background_path_or_data} (LIME formatado)")
            X_background_path = Path(X_background_path_or_data)
            if X_background_path.exists():
                X_background_data = np.load(X_background_path)
            else:
                logger.error(f"Arquivo de dados de fundo não encontrado: {X_background_path} (LIME formatado)")
                return 0, 0
        elif isinstance(X_background_path_or_data, np.ndarray):
            X_background_data = X_background_path_or_data
        else:
            logger.error("X_background_path_or_data deve ser um caminho ou np.ndarray (LIME formatado).")
            return 0, 0

        if X_background_data is None or X_background_data.size == 0:
            logger.error("Dados de fundo não disponíveis ou vazios para LIME formatado.")
            return 0, 0

        target_instance_normalized = None
        instance_source_info = ""
        random_idx_bg_fmt = -1

        if instance_to_explain_node is not None:
            if isinstance(instance_to_explain_node, np.ndarray):
                target_instance_normalized = instance_to_explain_node
                instance_source_info = "Instância específica fornecida (normalizada)."
                logger.info(f"LIME formatado: Usando instância específica fornecida. Forma: {target_instance_normalized.shape}")
            else:
                logger.error(f"LIME formatado: instance_to_explain_node fornecida, mas não é np.ndarray. Tipo: {type(instance_to_explain_node)}")
                return 0,0
        else:
            if X_background_data.shape[0] > 0:
                random_idx_bg_fmt = random.randint(0, X_background_data.shape[0] - 1)
                target_instance_normalized = X_background_data[random_idx_bg_fmt]
                instance_source_info = f"Instância aleatória (índice de fundo: {random_idx_bg_fmt}) dos dados de fundo."
                logger.info(f"LIME formatado: Nenhuma instância específica. Amostra aleatória {random_idx_bg_fmt} selecionada.")
            else:
                logger.error("LIME formatado: Dados de fundo vazios, não é possível selecionar instância.")
                return 0,0
        
        if target_instance_normalized is None:
            logger.error("LIME formatado: Não foi possível determinar a instância a ser explicada.")
            return 0,0

        if target_instance_normalized.ndim > 1:
            target_instance_normalized = target_instance_normalized.flatten()
        
        instance = target_instance_normalized

        prediction_output_raw = original_target_info.get('predicted_raw_value') if original_target_info else None
        if prediction_output_raw is None:
            if explainer.model and hasattr(explainer, 'device'):
                instance_tensor = torch.tensor(instance, dtype=torch.float32).to(explainer.device)
                with torch.no_grad():
                    prediction_output_raw = explainer.model(instance_tensor.unsqueeze(0)).item()
                logger.info(f"LIME formatado: Previsão raw calculada internamente: {prediction_output_raw:.4f}")
            else:
                prediction_output_raw = 0.0
                logger.warning("LIME formatado: Modelo ou dispositivo não configurado no explainer E previsão raw não fornecida. Previsão será 0.")
        else:
            logger.info(f"LIME formatado: Usando previsão raw fornecida via original_target_info: {prediction_output_raw:.4f}")


        instance_original_values_dict_fmt = None
        current_feature_names_for_desnorm = feature_names if feature_names is not None else getattr(explainer, 'feature_names', None)

        if scaler is not None and current_feature_names_for_desnorm is not None and instance is not None:
            try:
                instance_reshaped = instance.reshape(1, -1) if len(instance.shape) == 1 else instance
                if instance_reshaped.shape[1] == len(current_feature_names_for_desnorm):
                    desnormalized_values = scaler.inverse_transform(instance_reshaped)
                    instance_original_values_dict_fmt = dict(zip(current_feature_names_for_desnorm, desnormalized_values[0]))
                    for k_orig, v_orig in instance_original_values_dict_fmt.items():
                        if isinstance(v_orig, (float, np.floating)):
                            instance_original_values_dict_fmt[k_orig] = round(v_orig, 4)
                    logger.info("LIME formatado: Valores originais da instância obtidos via desnormalização com scaler.")
                else:
                    logger.warning(
                        f"LIME formatado: Disparidade no número de features para desnormalização com scaler. "
                        f"Instância tem {instance_reshaped.shape[1]} features, "
                        f"mas current_feature_names_for_desnorm tem {len(current_feature_names_for_desnorm)}."
                    )
            except Exception as e_desnorm:
                logger.error(f"LIME formatado: Erro ao tentar desnormalizar a instância com scaler: {e_desnorm}", exc_info=True)
        else:
            missing_items_desnorm = []
            if scaler is None: missing_items_desnorm.append("scaler")
            if current_feature_names_for_desnorm is None: missing_items_desnorm.append("feature_names (explainer ou argumento)")
            if instance is None: missing_items_desnorm.append("instance")
            logger.info(f"LIME formatado: Itens em falta para desnormalização com scaler: {', '.join(missing_items_desnorm)}.")


        final_feature_names_for_lime = None
        num_lime_features_to_use = 0
        
        num_features_in_instance = len(instance)
        if feature_names is not None and len(feature_names) == num_features_in_instance:
            final_feature_names_for_lime = feature_names
            num_lime_features_to_use = len(feature_names)
            logger.info(f"LIME formatado: Usando 'feature_names' (argumento) com {num_lime_features_to_use} features.")
        elif hasattr(explainer, 'feature_names') and explainer.feature_names is not None and len(explainer.feature_names) == num_features_in_instance:
            final_feature_names_for_lime = explainer.feature_names
            num_lime_features_to_use = len(explainer.feature_names)
            logger.info(f"LIME formatado: Usando 'explainer.feature_names' com {num_lime_features_to_use} features.")
        else:
            num_lime_features_to_use = num_features_in_instance
            final_feature_names_for_lime = [f"feature_{i}" for i in range(num_features_in_instance)]
            logger.warning(f"LIME formatado: Nomes de features (argumento ou explainer) não correspondem ao tamanho da instância ou não disponíveis. "
                           f"LIME usará num_features=len(instance)={num_lime_features_to_use} e nomes genéricos.")

        actual_lime_explainer = explainer.get_lime_explainer(X_background_data)
        if actual_lime_explainer is None:
            logger.error("Falha ao obter/inicializar o actual_lime_explainer a partir do ModelExplainer em generate_explainability_formatado. A explicação LIME não será gerada.")
            # Considerar como lidar com este erro - talvez retornar ou levantar uma exceção específica
            # Por agora, a função continuará e tentará escrever um ficheiro vazio ou parcial se lime_exp não for definido.
            # Idealmente, deveria retornar aqui para evitar mais erros.
            lime_exp = None # Garantir que lime_exp existe para o bloco finally, mas está vazio
        else:
            try:
                lime_exp = actual_lime_explainer.explain_instance(
                    data_row=instance, 
                    predict_fn=explainer.predict_fn, # Usar o predict_fn do ModelExplainer, que LIME espera
                    num_features=num_lime_features_to_use, 
                    labels=(1,)
                )
            except Exception as e_explain:
                logger.error(f"Erro durante a chamada a actual_lime_explainer.explain_instance: {e_explain}", exc_info=True)
                lime_exp = None

        if lime_exp is None:
            logger.error("A explicação LIME (lime_exp) não pôde ser gerada. O ficheiro de texto estará incompleto.")
            # Preencher com uma lista vazia para evitar erros no loop seguinte se lime_exp for None
            # e a lógica de escrita do ficheiro esperar um objeto com as_list()
            class EmptyLimeExp:
                def as_list(self):
                    return []
            lime_exp = EmptyLimeExp()
        
        logger.info(f"LIME_FORMATADO [{client_id_log}]: lime_exp.as_list() contém {len(lime_exp.as_list())} features para processar.")
        lime_txt_path = base_reports / 'lime_explanation_formatado.txt'
        os.makedirs(base_reports, exist_ok=True)

        with open(lime_txt_path, 'w') as f:
            f.write(f"Explicação LIME Formatada (Gerada em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n")
            f.write(f"Fonte da Instância: {instance_source_info}\n")
            if random_idx_bg_fmt != -1:
                 f.write(f"Índice da instância nos dados de fundo (X_background_data): {random_idx_bg_fmt}\n")

            predicted_class_display = "0 (Normal)" if prediction_output_raw <= 0.5 else "1 (Ataque)"
            f.write(f"Previsão Raw do Modelo para esta Instância: {prediction_output_raw:.4f} -> Classe Prevista: {predicted_class_display}\n")

            if original_target_info:
                f.write(f"Classe Original da Instância: {original_target_info.get('original_target_class', 'N/A')} (Valor Target: {original_target_info.get('original_target', 'N/A')})\n")
                correctness = "Correta" if original_target_info.get('prediction_correct', False) else "Incorreta"
                f.write(f"Previsão foi: {correctness}\n")
            else:
                f.write("(Informação do target original não fornecida)\n")

            f.write("\n--- Contribuições das Features (Ordenadas por Impacto Absoluto) ---\n")
            header = f"{'Feature':<30} | {'Valor Norm.':<15} | {'Valor Original':<20} | {'Impacto LIME':<20} | {'Interpretação'}"
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")

            temp_explanation_list = []
            
            # lime_exp.as_list() retorna (feature_index, weight)
            for feature_idx_lime, impact in lime_exp.as_list():
                # O feature_idx_lime é o índice na instância original, que corresponde ao final_feature_names_for_lime
                raw_feature_name = final_feature_names_for_lime[feature_idx_lime] if feature_idx_lime < len(final_feature_names_for_lime) else f"feature_unknown_idx_{feature_idx_lime}"
                processed_feature_name = extract_feature_name(raw_feature_name)
                 
                normalized_value = instance[feature_idx_lime] if feature_idx_lime < len(instance) else "N/A"
                
                original_value_display = "N/A"
                if instance_original_values_dict_fmt: # Desnormalizado via scaler
                    if processed_feature_name in instance_original_values_dict_fmt:
                        original_value_display = format_value(instance_original_values_dict_fmt[processed_feature_name])
                    elif raw_feature_name in instance_original_values_dict_fmt:
                        original_value_display = format_value(instance_original_values_dict_fmt[raw_feature_name])
                elif FEATURE_METADATA_AVAILABLE and hasattr(feature_metadata, 'denormalize_feature_value'): # Tentar com metadata
                    try:
                        denormalized_val_meta = feature_metadata.denormalize_feature_value(processed_feature_name, normalized_value)
                        if denormalized_val_meta is not None:
                             original_value_display = format_value(denormalized_val_meta)
                        # else:
                        #     logger.debug(f"Metadata denormalization retornou None para {processed_feature_name} com valor {normalized_value}")
                    except Exception as e_meta_denorm:
                        logger.warning(f"Erro ao desnormalizar '{processed_feature_name}' com feature_metadata: {e_meta_denorm}")

                interpretation = get_interpretation(impact)
                temp_explanation_list.append({
                    'raw_name': raw_feature_name,
                    'processed_name': processed_feature_name,
                    'norm_val': format_value(normalized_value) if normalized_value != "N/A" else "N/A",
                    'orig_val': original_value_display,
                    'impact': impact, 
                    'interpretation': interpretation
                })

            # DEBUG: Inspecionar temp_explanation_list e tipos de 'impact'
            logger.debug(f"LIME_FORMATADO [{client_id_log}]: temp_explanation_list ANTES da ordenação (total {len(temp_explanation_list)} itens):")
            for i, entry in enumerate(temp_explanation_list):
                impact_val = entry.get('impact')
                processed_name_val = entry.get('processed_name')
                logger.debug(f"  LIME_FORMATADO [{client_id_log}]: Item {i}: impact={impact_val} (type: {type(impact_val)}), name={processed_name_val}")
            logger.debug(f"LIME_FORMATADO [{client_id_log}]: Fim da inspeção de temp_explanation_list (ANTES da ordenação).")
            
            # Filtrar para garantir que apenas itens com 'impact' numérico são ordenados
            numeric_impact_list = [item for item in temp_explanation_list if isinstance(item.get('impact'), (int, float))]
            if len(numeric_impact_list) != len(temp_explanation_list):
                logger.warning(f"LIME formatado: {len(temp_explanation_list) - len(numeric_impact_list)} itens foram removidos da explicação devido a 'impact' não numérico.")

            sorted_explanation_list = sorted(numeric_impact_list, key=lambda x: abs(x['impact']), reverse=True)
            
            for item in sorted_explanation_list:
                line = f"{item['processed_name']:<30} | {item['norm_val']:<15} | {item['orig_val']:<20} | {format_value(item['impact']):<20} | {item['interpretation']}"
                f.write(line + "\n")

            logger.info(f'Explicação LIME (formatada) guardada em {lime_txt_path}')
    except BaseException as e_be: # Alterar de Exception para BaseException e o nome da variável
        # Tentar obter client_id_log se já definido, caso contrário usar um placeholder
        error_client_id_log = locals().get('client_id_log', getattr(explainer, 'cid', 'UNKNOWN_LIME_FORMAT_EXCEPT'))
        logger.error(f"LIME_FORMATADO [{error_client_id_log}]: Exceção BaseException capturada: {e_be}", exc_info=True)
        # Manter o bloco finally intacto por agora, pois ele calcula lime_duration
    finally:
        lime_duration = time.time() - lime_start
        final_client_id_log = locals().get('client_id_log', getattr(explainer, 'cid', 'UNKNOWN_LIME_FORMAT_FINALLY'))
        logger.info(f"LIME_FORMATADO [{final_client_id_log}]: Duração da formatação LIME para esta instância: {lime_duration:.2f}s")
    
    return lime_duration, shap_duration
