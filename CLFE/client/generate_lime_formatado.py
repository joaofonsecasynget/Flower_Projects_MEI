#!/usr/bin/env python
"""
Script simples para gerar manualmente o arquivo lime_explanation_formatado.txt
a partir do arquivo lime_explanation.txt.

Este script deve ser copiado para o contêiner do Docker se você 
quiser executá-lo dentro do contêiner.
"""

import os
import json
import logging
from pathlib import Path

# Configurar logging
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

def generate_lime_formatado(client_id=1):
    """
    Gera o arquivo lime_explanation_formatado.txt a partir do lime_explanation.txt.
    
    Args:
        client_id: ID do cliente para o qual gerar o arquivo (default: 1)
    """
    # Definir diretórios
    base_dir = Path('/app/client/reports') if os.path.exists('/app/client') else Path('reports')
    client_dir = base_dir / f'client_{client_id}'
    
    # Verificar se o diretório existe
    if not client_dir.exists():
        logger.error(f"Diretório {client_dir} não encontrado.")
        return False
    
    # Verificar se o arquivo lime_explanation.txt existe
    lime_explanation_file = client_dir / 'lime_explanation.txt'
    if not lime_explanation_file.exists():
        logger.error(f"Arquivo {lime_explanation_file} não encontrado.")
        return False
    
    # Verificar se o arquivo lime_instance_info.json existe
    lime_instance_info_file = client_dir / 'lime_instance_info.json'
    if not lime_instance_info_file.exists():
        logger.error(f"Arquivo {lime_instance_info_file} não encontrado.")
        return False
    
    try:
        # Ler o conteúdo do arquivo lime_explanation.txt
        with open(lime_explanation_file, 'r', encoding='utf-8') as f:
            lime_text = f.read()
        
        # Ler o conteúdo do arquivo lime_instance_info.json
        with open(lime_instance_info_file, 'r', encoding='utf-8') as f:
            instance_info = json.load(f)
        
        # Extrair informações
        try:
            exp_list = eval(lime_text)  # Converte a string para lista
            explained_features = {feat: impact for feat, impact in exp_list}
        except:
            logger.error(f"Erro ao converter lime_explanation.txt para lista.")
            explained_features = {}
        
        # Extrair informações da instância
        prediction = instance_info.get('prediction', 0.0)
        sample_idx = instance_info.get('index', 0)
        
        # Obter valores das features
        all_features = instance_info.get("features", {})
        
        # Criar o texto formatado
        formatted_explanation = ""
        formatted_explanation += "# Explicação LIME\n\n"
        
        # Informações da instância
        formatted_explanation += f"## Amostra {sample_idx}\n\n"
        formatted_explanation += "### Previsão do Modelo\n"
        formatted_explanation += f"- **Classe Prevista:** {'ATAQUE' if prediction >= 0.5 else 'NORMAL'}\n"
        formatted_explanation += f"- **Probabilidade:** {prediction:.4f}\n\n"
        
        # Categorizar features
        categories = {
            "Taxa de Download (dl_bitrate)": [],
            "Taxa de Upload (ul_bitrate)": [],
            "Bytes de Upload (não incrementais)": [],
            "Bytes de Download (não incrementais)": [],
            "Bytes de Upload (incrementais)": [],
            "Bytes de Download (incrementais)": [],
            "Pacotes de Upload": [],
            "Pacotes de Download": [],
            "Outras Features": []
        }
        
        # Preencher features por categoria
        for feature_name, value in all_features.items():
            # Obter contribuição (se existir)
            contribution = explained_features.get(feature_name, 0)
            
            if "dl_bitrate" in feature_name:
                categories["Taxa de Download (dl_bitrate)"].append((feature_name, value, contribution))
            elif "ul_bitrate" in feature_name:
                categories["Taxa de Upload (ul_bitrate)"].append((feature_name, value, contribution))
            elif "ul_total_bytes_non_incr" in feature_name:
                categories["Bytes de Upload (não incrementais)"].append((feature_name, value, contribution))
            elif "dl_total_bytes_non_incr" in feature_name:
                categories["Bytes de Download (não incrementais)"].append((feature_name, value, contribution))
            elif "ul_total_bytes_incr" in feature_name:
                categories["Bytes de Upload (incrementais)"].append((feature_name, value, contribution))
            elif "dl_total_bytes_incr" in feature_name:
                categories["Bytes de Download (incrementais)"].append((feature_name, value, contribution))
            elif "ul_packet" in feature_name:
                categories["Pacotes de Upload"].append((feature_name, value, contribution))
            elif "dl_packet" in feature_name:
                categories["Pacotes de Download"].append((feature_name, value, contribution))
            else:
                categories["Outras Features"].append((feature_name, value, contribution))
        
        # Ordenar features por impacto
        for category, features in categories.items():
            categories[category] = sorted(features, key=lambda x: abs(x[2]), reverse=True)
        
        # Contar estatísticas
        total_features = len(all_features)
        explained_features_count = len(exp_list) if isinstance(exp_list, list) else 0
        positive_impact = sum(1 for _, _, impact in [item for sublist in categories.values() for item in sublist] if impact > 0)
        negative_impact = sum(1 for _, _, impact in [item for sublist in categories.values() for item in sublist] if impact < 0)
        
        # Função para formatar valor
        def format_value(val):
            if isinstance(val, (int, float)):
                if float(val).is_integer():
                    return f"{int(float(val))}"
                else:
                    return f"{float(val):.4f}"
            else:
                return str(val)
        
        # Função para interpretar impacto
        def get_interpretation(impact):
            if impact > 0.001:
                return "Aumenta probabilidade de ATAQUE"
            elif impact < -0.001:
                return "Aumenta probabilidade de NORMAL"
            else:
                return "Neutro"
        
        # Adicionar cada categoria ao relatório
        for category, features in categories.items():
            if features:
                formatted_explanation += f"### {category}\n\n"
                
                # Preparar dados para a tabela
                feature_names = [f[0] for f in features]
                
                # Formatar valores
                values = [format_value(val) for _, val, _ in features]
                
                # Formatar contribuições
                contributions = []
                for _, _, impact in features:
                    if abs(impact) < 0.0001:
                        contributions.append("0")
                    else:
                        contributions.append(f"{impact:.6f}")
                
                interpretations = [get_interpretation(impact) for _, _, impact in features]
                
                # Calcular larguras de coluna
                max_feature_len = max(len(name) for name in feature_names) + 2
                max_value_len = max(len(val) for val in values) + 2
                max_contrib_len = max(len(contrib) for contrib in contributions) + 2
                max_interp_len = max(len(interp) for interp in interpretations) + 2
                
                # Gerar cabeçalho da tabela
                header = f"| {'Feature'.ljust(max_feature_len)} | {'Valor'.ljust(max_value_len)} | {'Contribuição'.ljust(max_contrib_len)} | {'Interpretação'.ljust(max_interp_len)} |"
                separator = f"|{'-' * (max_feature_len + 2)}|{'-' * (max_value_len + 2)}|{'-' * (max_contrib_len + 2)}|{'-' * (max_interp_len + 2)}|"
                
                formatted_explanation += header + "\n"
                formatted_explanation += separator + "\n"
                
                # Gerar linhas da tabela
                for i, (feature_name, value, impact) in enumerate(features):
                    value_str = format_value(value)
                    
                    if abs(impact) < 0.0001:
                        impact_str = "0"
                    else:
                        impact_str = f"{impact:.6f}"
                    
                    interpretation = get_interpretation(impact)
                    
                    row = f"| {feature_name.ljust(max_feature_len)} | {value_str.ljust(max_value_len)} | {impact_str.ljust(max_contrib_len)} | {interpretation.ljust(max_interp_len)} |"
                    formatted_explanation += row + "\n"
                
                formatted_explanation += "\n"
        
        # Adicionar estatísticas
        formatted_explanation += "## Estatísticas\n\n"
        formatted_explanation += f"- **Total de Features:** {total_features}\n"
        formatted_explanation += f"- **Features Explicadas pelo LIME:** {explained_features_count}\n"
        formatted_explanation += f"- **Features que aumentam probabilidade de ATAQUE:** {positive_impact}\n"
        formatted_explanation += f"- **Features que aumentam probabilidade de NORMAL:** {negative_impact}\n"
        formatted_explanation += f"- **Features neutras:** {total_features - positive_impact - negative_impact}\n\n"
        
        # Adicionar interpretação
        formatted_explanation += "## Interpretação\n\n"
        formatted_explanation += "- **Contribuições positivas (em vermelho):** Aumentam a probabilidade de ser classificado como ataque\n"
        formatted_explanation += "- **Contribuições negativas (em verde):** Aumentam a probabilidade de ser classificado como normal\n"
        formatted_explanation += "- **Contribuições neutras (valor 0):** Não têm influência na classificação\n"
        
        # Salvar o arquivo formatado
        lime_explanation_formatted_file = client_dir / "lime_explanation_formatado.txt"
        with open(lime_explanation_formatted_file, 'w', encoding='utf-8') as f:
            f.write(formatted_explanation)
        
        logger.info(f"Arquivo {lime_explanation_formatted_file} gerado com sucesso.")
        return True
    
    except Exception as e:
        logger.error(f"Erro ao gerar arquivo formatado: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    import sys
    
    # Obter ID do cliente a partir dos argumentos da linha de comando
    client_id = 1
    if len(sys.argv) > 1:
        try:
            client_id = int(sys.argv[1])
        except:
            pass
    
    # Gerar o arquivo formatado
    success = generate_lime_formatado(client_id)
    
    # Reportar resultado
    if success:
        print(f"Arquivo lime_explanation_formatado.txt gerado com sucesso para cliente {client_id}.")
    else:
        print(f"Falha ao gerar arquivo lime_explanation_formatado.txt para cliente {client_id}.")
