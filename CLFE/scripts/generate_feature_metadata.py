#!/usr/bin/env python
"""
Script para gerar e salvar metadados de features a partir do dataset original.
Isso cria um arquivo centralizado com informações sobre todas as features,
incluindo categorias, estatísticas e se são usadas para treinamento.
"""
import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np

# Adicionar diretório pai ao PYTHONPATH para importações relativas
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, parent_dir)

# Importar diretamente do módulo
sys.path.append(os.path.join(parent_dir, 'RLFE'))
from explainability.feature_metadata import feature_metadata

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Gera metadados de features a partir do dataset original')
    parser.add_argument('--dataset', type=str, required=True, help='Caminho para o arquivo CSV do dataset')
    parser.add_argument('--output', type=str, default=None, help='Caminho para salvar o arquivo de metadados')
    parser.add_argument('--summary', type=str, default=None, help='Caminho para salvar o resumo de metadados em formato texto')
    
    args = parser.parse_args()
    
    # Verificar se o dataset existe
    if not os.path.exists(args.dataset):
        logger.error(f"Dataset não encontrado: {args.dataset}")
        return 1
    
    # Definir caminho de saída se não especificado
    if not args.output:
        args.output = os.path.join(os.path.dirname(args.dataset), 'feature_metadata.json')
    
    # Carregar dataset
    logger.info(f"Carregando dataset de {args.dataset}")
    try:
        df = pd.read_csv(args.dataset)
        logger.info(f"Dataset carregado: {df.shape[0]} linhas, {df.shape[1]} colunas")
    except Exception as e:
        logger.error(f"Erro ao carregar dataset: {str(e)}")
        return 1
    
    # Configurar caminho do arquivo de metadados
    feature_metadata.metadata_path = args.output
    
    # Extrair metadados do dataset
    logger.info("Extraindo metadados do dataset")
    feature_metadata.extract_metadata_from_dataset(
        df, 
        normalize=True,
        remove_identifiers=True,
        extract_time_features=True
    )
    
    # Salvar metadados
    logger.info(f"Salvando metadados em {args.output}")
    if feature_metadata.save():
        logger.info("Metadados salvos com sucesso")
    else:
        logger.error("Erro ao salvar metadados")
        return 1
    
    # Gerar e salvar resumo, se solicitado
    if args.summary:
        logger.info(f"Gerando resumo em {args.summary}")
        summary_path = feature_metadata.export_summary(args.summary)
        if summary_path:
            logger.info(f"Resumo salvo em {summary_path}")
        else:
            logger.error("Erro ao salvar resumo")
    
    # Mostrar estatísticas básicas
    feature_count = len(feature_metadata.features)
    category_count = len(feature_metadata.categories)
    training_count = len(feature_metadata.get_training_features())
    
    logger.info(f"Total de features: {feature_count}")
    logger.info(f"Categorias identificadas: {category_count}")
    logger.info(f"Features para treinamento: {training_count}")
    
    logger.info("Processo concluído com sucesso")
    return 0

if __name__ == "__main__":
    sys.exit(main())
