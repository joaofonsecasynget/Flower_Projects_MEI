#!/usr/bin/env python
"""
Script para gerar manualmente o arquivo lime_explanation_formatado.txt
baseado nos arquivos existentes.
"""

import os
import sys
import logging
from pathlib import Path
import json

# Configurar logging
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

def fix_lime_explanation(reports_dir):
    """
    Cria o arquivo lime_explanation_formatado.txt a partir do lime_explanation.txt
    e do lime_instance_info.json existentes.
    
    Args:
        reports_dir: Diretório de relatórios do cliente
    """
    reports_path = Path(reports_dir)
    
    # Verificar se os arquivos necessários existem
    lime_explanation_path = reports_path / "lime_explanation.txt"
    lime_instance_info_path = reports_path / "lime_instance_info.json"
    
    if not lime_explanation_path.exists():
        logger.error(f"Arquivo lime_explanation.txt não encontrado em {lime_explanation_path}")
        return False
    
    if not lime_instance_info_path.exists():
        logger.error(f"Arquivo lime_instance_info.json não encontrado em {lime_instance_info_path}")
        return False
    
    try:
        # Ler o arquivo lime_explanation.txt
        with open(lime_explanation_path, 'r', encoding='utf-8') as f:
            lime_text = f.read()
        
        # Ler informações da instância
        with open(lime_instance_info_path, 'r', encoding='utf-8') as f:
            instance_info = json.load(f)
        
        # Extrair informações do lime_explanation.txt
        lime_exp_list = eval(lime_text)  # Converte a string para lista
        explained_features = {feat: impact for feat, impact in lime_exp_list}
        
        # Obter valores das features da instância
        all_features = instance_info.get("features", {})
        
        # Formatar a explicação
        formatted_explanation = ""
        formatted_explanation += "# Explicação LIME\n\n"
        
        # Informações da instância
        formatted_explanation += f"## Amostra {instance_info.get('index', 'N/A')}\n\n"
        formatted_explanation += "### Previsão do Modelo\n"
        formatted_explanation += f"- **Classe Prevista:** {'ATAQUE' if instance_info.get('prediction', 0) >= 0.5 else 'NORMAL'}\n"
        formatted_explanation += f"- **Probabilidade:** {instance_info.get('prediction', 0):.4f}\n\n"
        
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
        
        # Ordenar por impacto absoluto
        for category, features in categories.items():
            categories[category] = sorted(features, key=lambda x: abs(x[2]), reverse=True)
        
        # Contar estatísticas
        total_features = len(all_features)
        explained_features_count = len(lime_exp_list)
        positive_impact = sum(1 for _, _, impact in [item for sublist in categories.values() for item in sublist] if impact > 0)
        negative_impact = sum(1 for _, _, impact in [item for sublist in categories.values() for item in sublist] if impact < 0)
        
        # Função para formatar valor de acordo com seu tipo
        def format_value(val):
            # Se o valor for inteiro, remover casas decimais
            if isinstance(val, (int, float)) and float(val).is_integer():
                return f"{int(float(val))}"
            elif isinstance(val, float):
                return f"{val:.4f}"  # Limitar a 4 casas decimais
            else:
                return str(val)
        
        # Função para retornar uma interpretação textual com base no valor de impacto
        def get_interpretation(impact):
            if impact > 0.001:  # Limite de significância
                return "Aumenta probabilidade de ATAQUE"
            elif impact < -0.001:  # Limite de significância
                return "Aumenta probabilidade de NORMAL"
            else:
                return "Neutro"
        
        # Adicionar cada categoria ao relatório formatado
        for category, features in categories.items():
            if features:  # Só adicionar categorias que têm features
                formatted_explanation += f"### {category}\n\n"
                
                # Calcular colunas para alinhamento adequado
                feature_names = [f[0] for f in features]
                
                # Formatar valores (remover casas decimais se for inteiro)
                values = []
                for _, val, _ in features:
                    values.append(format_value(val))
                
                # Formatar contribuições
                contributions = []
                for _, _, impact in features:
                    if abs(impact) < 0.0001:  # Limiar para considerar zero
                        contributions.append("0")
                    else:
                        contributions.append(f"{impact:.6f}")
                
                interpretations = [get_interpretation(f[2]) for f in features]
                
                # Determinar largura máxima para cada coluna (com padding)
                max_feature_len = max(len(name) for name in feature_names) + 2
                max_value_len = max(len(val) for val in values) + 2
                max_contrib_len = max(len(contrib) for contrib in contributions) + 2
                max_interp_len = max(len(interp) for interp in interpretations) + 2
                
                # Cabeçalho da tabela com larguras adequadas
                header = f"| {'Feature'.ljust(max_feature_len)} | {'Valor'.ljust(max_value_len)} | {'Contribuição'.ljust(max_contrib_len)} | {'Interpretação'.ljust(max_interp_len)} |"
                separator = f"|{'-' * (max_feature_len + 2)}|{'-' * (max_value_len + 2)}|{'-' * (max_contrib_len + 2)}|{'-' * (max_interp_len + 2)}|"
                
                formatted_explanation += header + "\n"
                formatted_explanation += separator + "\n"
                
                # Linhas da tabela
                for i, (feature_name, value, impact) in enumerate(features):
                    value_str = format_value(value)
                    
                    # Formatação da contribuição
                    if abs(impact) < 0.0001:  # Limiar para considerar zero
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
        lime_explanation_formatted_file = reports_path / "lime_explanation_formatado.txt"
        with open(lime_explanation_formatted_file, 'w', encoding='utf-8') as f:
            f.write(formatted_explanation)
        
        logger.info(f"Arquivo lime_explanation_formatado.txt criado com sucesso em {lime_explanation_formatted_file}")
        return True
    
    except Exception as e:
        logger.error(f"Erro ao processar arquivos LIME: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python fix_lime_formatado.py <reports_directory>")
        print("Exemplo: python fix_lime_formatado.py /Users/joaofonseca/git/Flower_Projects_MEI/CLFE/client/reports/client_1")
        sys.exit(1)
    
    reports_dir = sys.argv[1]
    if not os.path.isdir(reports_dir):
        print(f"Erro: O diretório {reports_dir} não existe.")
        sys.exit(1)
    
    success = fix_lime_explanation(reports_dir)
    sys.exit(0 if success else 1)
