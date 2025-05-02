"""
Funções para geração de explicações LIME e SHAP para instâncias específicas.
"""
import os
import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import lime
import lime.lime_tabular
import shap

from .feature_utils import get_feature_category
from .shap_visualizations import gerar_tabela_shap
from .html_report import generate_html_report

# Configurar logger
logger = logging.getLogger("explainability.explainers")

def explain_prediction(model, instance, X_train, feature_names, feature_map, args, original_instance=None):
    """
    Gera explicações para a previsão do modelo para uma instância específica.
    
    Args:
        model: Modelo treinado
        instance: Instância a ser explicada no formato que o modelo espera
        X_train: Dados de treinamento para referência
        feature_names: Nomes das features
        feature_map: Mapeamento de features
        args: Argumentos da linha de comando
        original_instance: Valores originais da instância (antes de normalização)
    """
    logger.info("Gerando explicações para a instância selecionada")
    
    # Diretório para salvar explicações
    if hasattr(args, 'output_dir') and args.output_dir:
        output_dir = args.output_dir
    elif hasattr(args, 'output') and args.output:
        # Diretório baseado no argumento output
        if hasattr(args, 'instance_idx') and args.instance_idx is not None:
            instance_desc = f"instance_{args.instance_idx}"
        elif hasattr(args, 'random') and args.random:
            instance_desc = "instance_random"
        else:
            instance_desc = "instance"
        
        output_dir = f"{args.output}/client_{args.client}/{instance_desc}"
    else:
        # Diretório padrão
        if hasattr(args, 'instance_idx') and args.instance_idx is not None:
            instance_desc = f"instance_{args.instance_idx}"
        elif hasattr(args, 'random') and args.random:
            instance_desc = "instance_random"
        else:
            instance_desc = "instance"
        
        output_dir = f"RLFE/instance_explanations/client_{args.client}/{instance_desc}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Origem dos dados - dataset original ou dados do cliente
    dataset_origin = "Dataset original completo" if hasattr(args, '_use_original') and args._use_original else "dados de treinamento do cliente"
    
    # Converter instância para float32 para evitar problemas de compatibilidade
    instance = instance.astype(np.float32)
    
    # Fazer previsão com o modelo
    with torch.no_grad():
        input_tensor = torch.from_numpy(instance).float().reshape(1, -1)
        prediction = model(input_tensor).item()
    
    logger.info(f"Previsão para a instância: {prediction:f}")
    class_label = "Normal" if prediction < 0.5 else "Ataque"
    logger.info(f"Classe prevista: {class_label} (limiar = 0.5)")
    
    # Salvar informações básicas da previsão
    with open(os.path.join(output_dir, "prediction_info.txt"), "w") as f:
        f.write(f"Previsão: {prediction}\n")
        f.write(f"Classe: {class_label}\n")
        f.write(f"Cliente: {args.client}\n")
        
        # Descrever a origem do índice
        if hasattr(args, 'instance_idx') and args.instance_idx is not None:
            idx_desc = f"Fixo (índice {args.instance_idx})"
        elif hasattr(args, 'random') and args.random:
            idx_desc = f"Aleatório (índice {args._instance_idx})"
        else:
            idx_desc = "Não especificado"
        
        f.write(f"Índice da instância: {idx_desc}")
        
        # Adicionar índice original, se disponível
        if hasattr(args, '_original_dataset_index') and args._original_dataset_index is not None:
            f.write(f" (índice original: {args._original_dataset_index})")
        
        f.write("\n")
        f.write(f"Origem dos dados: {dataset_origin}\n")
    
    # Salvar arquivo com todos os valores da instância
    with open(os.path.join(output_dir, "instance_values_full.txt"), 'w') as f:
        f.write("Valores completos da instância analisada:\n\n")
        for i, (name, value) in enumerate(zip(feature_names, instance)):
            # Verificar se o valor é um número inteiro
            if value == int(value):
                formatted_value = f"{int(value)}"
            else:
                formatted_value = f"{value:.6f}"
            f.write(f"{i+1:4d}. {name:30s}: {formatted_value}\n")
    
    # Criar explicações LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        mode="regression"
    )
    
    # Função para fazer previsões
    def predict_fn(x):
        with torch.no_grad():
            input_tensor = torch.from_numpy(x).float()
            return model(input_tensor).numpy()
    
    # Explicação LIME
    lime_exp = explainer.explain_instance(
        instance,
        predict_fn,
        num_features=10
    )
    
    # Salvar explicação LIME em HTML
    lime_exp.save_to_file(os.path.join(output_dir, "lime_explanation.html"))
    
    # Salvar visualização LIME
    plt.figure(figsize=(10, 6))
    lime_exp.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "lime_explanation.png"))
    plt.close()
    
    # Explicação SHAP
    logger.info("Gerando valores SHAP (usando 100 amostras de background)")
    
    # Usar os valores originais se disponíveis
    instance_values_for_display = original_instance if original_instance is not None else instance
    
    # Sampler para o SHAP
    background = shap.sample(X_train, min(100, len(X_train)))
    
    # Definir função de previsão para SHAP
    def model_predict(x):
        with torch.no_grad():
            tensor_x = torch.tensor(x, dtype=torch.float32)
            return model(tensor_x).numpy()
    
    # Usar KernelExplainer
    shap_explainer = shap.KernelExplainer(model_predict, background)
    shap_values = shap_explainer.shap_values(instance.reshape(1, -1))
    
    # Processar valores SHAP
    if isinstance(shap_values, list):
        shap_values_instance = shap_values[0]
    else:
        shap_values_instance = shap_values.reshape(-1)
    
    # Associar os valores SHAP com as features e seus valores
    shap_data = []
    for i, (name, value, shap_value) in enumerate(zip(feature_names, instance_values_for_display, shap_values_instance)):
        category = get_feature_category(name)
        shap_data.append({
            "feature": name,
            "value": value,
            "shap_value": shap_value,
            "abs_shap_value": abs(shap_value),
            "category": category
        })
    
    # Ordenar por importância (valor absoluto)
    shap_data = sorted(shap_data, key=lambda x: x["abs_shap_value"], reverse=True)
    
    # Salvar em CSV
    shap_df = pd.DataFrame(shap_data)
    shap_df.columns = ["Feature", "Feature Value", "SHAP Value", "Absolute Value", "Category"]
    shap_df.to_csv(os.path.join(output_dir, "shap_values.csv"), index=False)
    
    # Gerar tabela textual SHAP
    with open(os.path.join(output_dir, "shap_explanation.txt"), "w") as f:
        f.write("TOP 10 FEATURES POR IMPACTO (SHAP)\n")
        f.write("================================\n\n")
        
        for i, item in enumerate(shap_data[:10]):
            impact = "aumenta" if item["shap_value"] > 0 else "diminui"
            f.write(f"{i+1}. {item['feature']} = {item['value']}\n")
            f.write(f"   SHAP value: {item['shap_value']:.6f}\n")
            f.write(f"   Impacto: {impact} a predição\n\n")
    
    # Criar resumo de explicabilidade
    with open(os.path.join(output_dir, "explanation_summary.txt"), "w") as f:
        f.write("RESUMO DA EXPLICABILIDADE\n")
        f.write("========================\n\n")
        f.write(f"Previsão: {prediction}\n")
        f.write(f"Classe prevista: {class_label}\n\n")
        
        f.write("Top 5 Features (LIME):\n")
        lime_features = lime_exp.as_list()[:5]
        for i, (feature, value) in enumerate(lime_features):
            f.write(f"{i+1}. {feature}: {value:.4f}\n")
        
        f.write("\nTop 5 Features (SHAP):\n")
        for i, item in enumerate(shap_data[:5]):
            f.write(f"{i+1}. {item['feature']}: {item['shap_value']:.4f}\n")
    
    # Agrupar features por categoria para o relatório HTML
    category_values = {}
    for item in shap_data:
        category = item["category"]
        if category not in category_values:
            category_values[category] = []
        
        category_values[category].append({
            "feature": item["feature"],
            "value": item["value"],
            "shap_value": item["shap_value"]
        })
    
    # Gerar relatório HTML consolidado
    generate_html_report(
        output_dir,
        prediction,
        class_label,
        args.client,
        idx_desc,
        args._original_dataset_index if hasattr(args, '_original_dataset_index') else None,
        lime_exp,
        feature_names,
        instance_values_for_display,
        shap_values_instance,
        expected_value=shap_explainer.expected_value,
        shap_table_html=gerar_tabela_shap(shap_values_instance, feature_names, instance_values_for_display, category_values),
        category_values=category_values,
        dataset_origin=dataset_origin,
        target_value=args._target_value if hasattr(args, '_target_value') else None
    )
    
    logger.info(f"Explicações geradas e salvas em: {output_dir}")
    return output_dir

def explain_with_shap(model, instance, feature_names, X_train, args=None, instance_features=None):
    """
    Gera explicações baseadas em SHAP para a instância fornecida.
    
    Args:
        model: Modelo treinado
        instance: Instância para explicar
        feature_names: Nomes das features
        X_train: Dados de treinamento para background distribution
        args: Argumentos da linha de comando
        instance_features: Valores originais das features (se disponível)
    
    Returns:
        Tuple: (shap_values, valores da instância, explicação textual)
    """
    logger.info(f"Gerando valores SHAP (usando {min(len(X_train), 100)} amostras de background)")
    background = shap.sample(X_train, min(len(X_train), 100))
    
    # Usar o explainer adequado
    explainer = shap.KernelExplainer(model.forward_numpy, background)
    shap_values = explainer.shap_values(instance.reshape(1, -1))
    
    # Preparar dados para visualização
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values[0],
        'Absolute Value': np.abs(shap_values[0]),
        'Feature Value': instance if instance_features is None else instance_features,
        'Category': [get_feature_category(feature) for feature in feature_names]
    })
    
    # Ordenar por importância (valor absoluto)
    feature_importance = feature_importance.sort_values('Absolute Value', ascending=False)
    
    # Gerar explicação em texto
    explanation_text = "TOP 10 FEATURES POR IMPACTO (SHAP)\n"
    explanation_text += "================================\n\n"
    
    top_features = feature_importance.head(10)
    for i, (_, row) in enumerate(top_features.iterrows()):
        impact_direction = "aumenta" if row['SHAP Value'] > 0 else "diminui"
        explanation_text += f"{i+1}. {row['Feature']} = {row['Feature Value']}\n"
        explanation_text += f"   SHAP value: {row['SHAP Value']:.6f}\n"
        explanation_text += f"   Impacto: {impact_direction} a predição\n\n"
    
    return shap_values[0], feature_importance, explanation_text
