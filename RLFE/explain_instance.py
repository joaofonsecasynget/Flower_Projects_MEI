#!/usr/bin/env python3
"""
Script para gerar explicações para instâncias específicas de um modelo de federated learning.
Este script foi refatorado para melhor organização e manutenção.
"""
import os
import sys
import random
import logging
import argparse
import numpy as np
import pandas as pd
import torch

from explainability.data_loader import load_model, load_train_data, select_instance
from explainability.feature_utils import extract_feature_names
from explainability.explainers import explain_prediction

# Configurar logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger("explain_instance")

def parse_args():
    """Configura e processa os argumentos de linha de comando"""
    parser = argparse.ArgumentParser(description='Explicar previsão para uma instância específica')
    
    # Argumentos obrigatórios
    parser.add_argument('--client', type=int, required=True, help='ID do cliente (ex: 1 para client_1)')
    
    # Forma de selecionar a instância (por índice ou por filtro)
    instance_group = parser.add_mutually_exclusive_group(required=True)
    instance_group.add_argument('--index', type=int, help='Índice da instância no dataset original')
    instance_group.add_argument('--random', action='store_true', help='Selecionar uma instância aleatória')
    
    # Argumentos opcionais
    parser.add_argument('--dataset', type=str, default="RLFE/DatasetIOT/transformed_dataset_imeisv_8642840401612300.csv", 
                        help='Caminho para o dataset original (CSV)')
    parser.add_argument('--output', type=str, default='RLFE/instance_explanations', 
                        help='Diretório para salvar os resultados')
    parser.add_argument('--seed', type=int, default=42, help='Seed para reprodutibilidade')
    parser.add_argument('--shap-samples', type=int, default=100, 
                        help='Número de amostras para o explainer SHAP (mais = mais preciso, mas mais lento)')
    parser.add_argument('--target-col', type=str, default="attack", 
                        help='Nome da coluna target no dataset')
    parser.add_argument('--use-original-dataset', action='store_true', 
                        help='(NOTA: Ignorado se o dataset original não estiver disponível) Usar o dataset original completo para seleção de instância')
    
    return parser.parse_args()

def main(args):
    """Função principal para explicação de instâncias"""
    
    # Definir seed para reprodutibilidade
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    
    # Carregar o modelo treinado e os dados
    model, X_train = load_model(args.client)
    
    # Carregar o dataset original se fornecido
    if args.use_original_dataset:
        X_data, is_original_dataset, original_indices, target_values, feature_columns = load_train_data(args.client, args.dataset)
        dataset_desc = "dataset original completo"
    else:
        X_data, is_original_dataset, original_indices, target_values, feature_columns = X_train, False, None, None, None
        dataset_desc = "dados de treinamento do cliente"
    
    logger.info(f"Usando {dataset_desc} para seleção de instância")
    
    # Selecionar uma instância para explicação
    instance_idx = select_instance(X_data, args, is_original_dataset)
    instance = X_data[instance_idx]
    
    # Armazenar os valores originais da instância para o relatório
    original_instance = np.copy(instance)
    
    # Se temos índices originais, armazenar para exibição
    if original_indices is not None and instance_idx < len(original_indices):
        args._original_dataset_index = original_indices[instance_idx]
        logger.info(f"Índice no dataset original: {args._original_dataset_index}")
    
    # Se temos valores target, armazenar para comparação
    if target_values is not None and instance_idx < len(target_values):
        args._target_value = target_values[instance_idx]
        logger.info(f"Valor target real: {args._target_value}")
    
    # Obter nomes reais das features baseado em múltiplas estratégias
    feature_names, feature_map = extract_feature_names(
        args.client, 
        X_data.shape[1], 
        instance,
        args.dataset if is_original_dataset else None,
        is_original_dataset,
        feature_columns
    )
    
    # Gerar explicações
    args._instance_idx = instance_idx
    args._use_original = is_original_dataset
    
    output_dir = explain_prediction(
        model, 
        instance, 
        X_train, 
        feature_names, 
        feature_map, 
        args,
        original_instance  # Passar os valores originais da instância
    )
    
    # Mostrar instruções finais
    logger.info("\n" + "="*80)
    logger.info(f"RESUMO DE EXECUÇÃO")
    logger.info("="*80)
    logger.info(f"Cliente analisado: {args.client}")
    logger.info(f"Instância analisada: {instance_idx} " + 
                f"({'Índice específico' if args.index is not None else 'Seleção aleatória'})")
    if hasattr(args, '_original_dataset_index'):
        logger.info(f"Índice no dataset original: {args._original_dataset_index}")
    logger.info(f"Todos os resultados foram salvos em: {output_dir}")
    logger.info(f"Arquivos gerados:")
    logger.info(f"  - prediction_info.txt: Informações básicas da previsão")
    logger.info(f"  - lime_explanation.html: Explicação LIME interativa (abrir em navegador)")
    logger.info(f"  - lime_explanation.png: Visualização estática LIME")
    logger.info(f"  - shap_values.csv: Valores SHAP para a instância")
    logger.info(f"  - shap_explanation.txt: Explicação SHAP em texto")
    logger.info(f"  - explanation_summary.txt: Resumo textual da explicabilidade")
    logger.info(f"  - instance_report.html: Relatório consolidado de explicabilidade da instância")
    logger.info("="*80)
    
    return output_dir

if __name__ == "__main__":
    args = parse_args()
    main(args)
