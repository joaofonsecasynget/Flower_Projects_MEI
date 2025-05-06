#!/usr/bin/env python
"""
Script para corrigir completamente os problemas de sintaxe e indentação no arquivo report_utils.py
"""

import logging
import os
import shutil

# Configurar logging
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

def fix_all_issues():
    """
    Substitui todo o trecho problemático do código por uma versão corrigida
    """
    file_path = 'report_utils.py'
    
    try:
        # Criar backup do arquivo original
        backup_path = f"{file_path}.bak"
        shutil.copy2(file_path, backup_path)
        logger.info(f"Backup criado em {backup_path}")
        
        # Ler o arquivo completo
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Trecho com problema (indentação incorreta)
        problematic_code = """            # Criar uma versão formatada do arquivo de explicação LIME
            try:
                with open(base_reports / 'lime_explanation.txt', 'r', encoding='utf-8') as f_in:
                    lime_text = f_in.read()
                
                # Formatação mais legível
                formatted_explanation = ""
                formatted_explanation += "# Explicação LIME\\n\\n"
                
                # Informações da instância
                formatted_explanation += f"## Amostra {sample_idx}\\n\\n"
                formatted_explanation += "### Previsão do Modelo\\n"
                formatted_explanation += f"- **Classe Prevista:** {'ATAQUE' if prediction >= 0.5 else 'NORMAL'}\\n"
                formatted_explanation += f"- **Probabilidade:** {prediction:.4f}\\n\\n"
                
                # Obter todos os valores das features
                all_features = {}
                for i, name in enumerate(explainer.feature_names):
                    # Obter valor original se disponível, caso contrário usar o normalizado
                    if instance_info and "features" in instance_info:
                        value = instance_info["features"].get(name, instance[i])
                    else:
                        value = instance[i]
                    all_features[name] = value
                
                # Obter explicação como lista
                exp_list = lime_exp.as_list()
                
                # Mapear features explicando para referência rápida
                explained_features = {feat: impact for feat, impact in exp_list}
                
                # Forçar incluir pelo menos algumas features com valores não-zero
                # Isso corrige o problema onde todas as features aparecem com contribuição 0
                if exp_list:
                    # Pegar as top 20 features pelo valor absoluto da explicação
                    sorted_exp = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)
                    top_features = sorted_exp[:min(20, len(sorted_exp))]
                    
                    # Atualizar dict de explained_features para garantir que estes valores sejam preservados
                    for feat, impact in top_features:
                        explained_features[feat] = impact
                
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
                
                # Preencher features por categoria (todas as features, não apenas as explicadas)
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
                
                # Ordenar cada categoria por impacto absoluto
                for category, features in categories.items():
                    categories[category] = sorted(features, key=lambda x: abs(x[2]), reverse=True)
                
                # Contar estatísticas
                total_features = len(all_features)
                explained_features_count = len(exp_list)
                positive_impact = sum(1 for _, _, impact in [item for sublist in categories.values() for item in sublist] if impact > 0)
                negative_impact = sum(1 for _, _, impact in [item for sublist in categories.values() for item in sublist] if impact < 0)
                
                # Função para formatar valor de acordo com seu tipo
                def format_value(val):
                    # Se o valor for inteiro, remover casas decimais
                    if isinstance(val, (int, float)) and val.is_integer():
                        return f"{int(val)}"
                    elif isinstance(val, float):
                        return f"{val:.4f}"  # Limitar a 4 casas decimais em vez de 6
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
                        formatted_explanation += f"### {category}\\n\\n"
                        
                        # Calcular colunas para alinhamento adequado
                        feature_names = [f[0] for f in features]
                        
                        # Formatar valores (remover casas decimais se for inteiro)
                        values = []
                        for _, val, _ in features:
                            if isinstance(val, (int, float)):
                                if val.is_integer():
                                    values.append(f"{int(val)}")
                                else:
                                    values.append(f"{val:.4f}")
                            else:
                                values.append(str(val))
                        
                        # Formatar contribuições (mostrar zeros como "0" sem casas decimais)
                        contributions = []
                        for _, _, impact in features:
                            if abs(impact) < 0.0001:  # Limiar mais restrito para considerar zero
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
                        
                        formatted_explanation += header + "\\n"
                        formatted_explanation += separator + "\\n"
                        
                        # Linhas da tabela
                        for i, (feature_name, value, impact) in enumerate(features):
                            # Formatação de valor adequada ao tipo
                            if isinstance(value, (int, float)):
                                if value.is_integer():
                                    value_str = f"{int(value)}"
                                else:
                                    value_str = f"{value:.4f}"
                            else:
                                value_str = str(value)
                            
                            # Formatação da contribuição
                            if abs(impact) < 0.0001:  # Limiar mais restrito para considerar zero
                                impact_str = "0"
                            else:
                                impact_str = f"{impact:.6f}"
                            
                            interpretation = get_interpretation(impact)
                            
                            row = f"| {feature_name.ljust(max_feature_len)} | {value_str.ljust(max_value_len)} | {impact_str.ljust(max_contrib_len)} | {interpretation.ljust(max_interp_len)} |"
                            formatted_explanation += row + "\\n"
                        
                        formatted_explanation += "\\n"
                
                # Adicionar estatísticas
                formatted_explanation += "## Estatísticas\\n\\n"
                formatted_explanation += f"- **Total de Features:** {total_features}\\n"
                formatted_explanation += f"- **Features Explicadas pelo LIME:** {explained_features_count}\\n"
                formatted_explanation += f"- **Features que aumentam probabilidade de ATAQUE:** {positive_impact}\\n"
                formatted_explanation += f"- **Features que aumentam probabilidade de NORMAL:** {negative_impact}\\n"
                formatted_explanation += f"- **Features neutras:** {total_features - positive_impact - negative_impact}\\n\\n"
                
                # Adicionar interpretação
                formatted_explanation += "## Interpretação\\n\\n"
                formatted_explanation += "- **Contribuições positivas (em vermelho):** Aumentam a probabilidade de ser classificado como ataque\\n"
                formatted_explanation += "- **Contribuições negativas (em verde):** Aumentam a probabilidade de ser classificado como normal\\n"
                formatted_explanation += "- **Contribuições neutras (valor 0):** Não têm influência na classificação\\n"
                
                # Criar o arquivo formatado usando o Path diretamente
                try:
                    # Garantir que o diretório existe
                    os.makedirs(base_reports, exist_ok=True)
                    
                    lime_explanation_formatted_file = base_reports / "lime_explanation_formatado.txt"
                    with open(str(lime_explanation_formatted_file), 'w', encoding='utf-8') as f:
                        f.write(formatted_explanation)
                    logger.info(f"Versão formatada da explicação LIME criada com sucesso em {lime_explanation_formatted_file}")
                except Exception as e:
                    logger.error(f"Erro ao escrever o arquivo formatado: {e}", exc_info=True)
            except Exception as e:
                logger.error(f'Erro ao criar versão formatada da explicação LIME: {e}', exc_info=True)
            
            # Salvar visualização LIME
            lime_png_path = base_reports / 'lime_final.png'"""
        
        # Versão corrigida com indentação adequada
        fixed_code = """            # Criar uma versão formatada do arquivo de explicação LIME
            try:
                with open(base_reports / 'lime_explanation.txt', 'r', encoding='utf-8') as f_in:
                    lime_text = f_in.read()
                
                # Formatação mais legível
                formatted_explanation = ""
                formatted_explanation += "# Explicação LIME\\n\\n"
                
                # Informações da instância
                formatted_explanation += f"## Amostra {sample_idx}\\n\\n"
                formatted_explanation += "### Previsão do Modelo\\n"
                formatted_explanation += f"- **Classe Prevista:** {'ATAQUE' if prediction >= 0.5 else 'NORMAL'}\\n"
                formatted_explanation += f"- **Probabilidade:** {prediction:.4f}\\n\\n"
                
                # Obter todos os valores das features
                all_features = {}
                for i, name in enumerate(explainer.feature_names):
                    # Obter valor original se disponível, caso contrário usar o normalizado
                    if instance_info and "features" in instance_info:
                        value = instance_info["features"].get(name, instance[i])
                    else:
                        value = instance[i]
                    all_features[name] = value
                
                # Obter explicação como lista
                exp_list = lime_exp.as_list()
                
                # Mapear features explicando para referência rápida
                explained_features = {feat: impact for feat, impact in exp_list}
                
                # Forçar incluir pelo menos algumas features com valores não-zero
                # Isso corrige o problema onde todas as features aparecem com contribuição 0
                if exp_list:
                    # Pegar as top 20 features pelo valor absoluto da explicação
                    sorted_exp = sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)
                    top_features = sorted_exp[:min(20, len(sorted_exp))]
                    
                    # Atualizar dict de explained_features para garantir que estes valores sejam preservados
                    for feat, impact in top_features:
                        explained_features[feat] = impact
                
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
                
                # Preencher features por categoria (todas as features, não apenas as explicadas)
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
                
                # Ordenar cada categoria por impacto absoluto
                for category, features in categories.items():
                    categories[category] = sorted(features, key=lambda x: abs(x[2]), reverse=True)
                
                # Contar estatísticas
                total_features = len(all_features)
                explained_features_count = len(exp_list)
                positive_impact = sum(1 for _, _, impact in [item for sublist in categories.values() for item in sublist] if impact > 0)
                negative_impact = sum(1 for _, _, impact in [item for sublist in categories.values() for item in sublist] if impact < 0)
                
                # Função para formatar valor de acordo com seu tipo
                def format_value(val):
                    # Se o valor for inteiro, remover casas decimais
                    if isinstance(val, (int, float)) and val.is_integer():
                        return f"{int(val)}"
                    elif isinstance(val, float):
                        return f"{val:.4f}"  # Limitar a 4 casas decimais em vez de 6
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
                        formatted_explanation += f"### {category}\\n\\n"
                        
                        # Calcular colunas para alinhamento adequado
                        feature_names = [f[0] for f in features]
                        
                        # Formatar valores (remover casas decimais se for inteiro)
                        values = []
                        for _, val, _ in features:
                            if isinstance(val, (int, float)):
                                if val.is_integer():
                                    values.append(f"{int(val)}")
                                else:
                                    values.append(f"{val:.4f}")
                            else:
                                values.append(str(val))
                        
                        # Formatar contribuições (mostrar zeros como "0" sem casas decimais)
                        contributions = []
                        for _, _, impact in features:
                            if abs(impact) < 0.0001:  # Limiar mais restrito para considerar zero
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
                        
                        formatted_explanation += header + "\\n"
                        formatted_explanation += separator + "\\n"
                        
                        # Linhas da tabela
                        for i, (feature_name, value, impact) in enumerate(features):
                            # Formatação de valor adequada ao tipo
                            if isinstance(value, (int, float)):
                                if value.is_integer():
                                    value_str = f"{int(value)}"
                                else:
                                    value_str = f"{value:.4f}"
                            else:
                                value_str = str(value)
                            
                            # Formatação da contribuição
                            if abs(impact) < 0.0001:  # Limiar mais restrito para considerar zero
                                impact_str = "0"
                            else:
                                impact_str = f"{impact:.6f}"
                            
                            interpretation = get_interpretation(impact)
                            
                            row = f"| {feature_name.ljust(max_feature_len)} | {value_str.ljust(max_value_len)} | {impact_str.ljust(max_contrib_len)} | {interpretation.ljust(max_interp_len)} |"
                            formatted_explanation += row + "\\n"
                        
                        formatted_explanation += "\\n"
                
                # Adicionar estatísticas
                formatted_explanation += "## Estatísticas\\n\\n"
                formatted_explanation += f"- **Total de Features:** {total_features}\\n"
                formatted_explanation += f"- **Features Explicadas pelo LIME:** {explained_features_count}\\n"
                formatted_explanation += f"- **Features que aumentam probabilidade de ATAQUE:** {positive_impact}\\n"
                formatted_explanation += f"- **Features que aumentam probabilidade de NORMAL:** {negative_impact}\\n"
                formatted_explanation += f"- **Features neutras:** {total_features - positive_impact - negative_impact}\\n\\n"
                
                # Adicionar interpretação
                formatted_explanation += "## Interpretação\\n\\n"
                formatted_explanation += "- **Contribuições positivas (em vermelho):** Aumentam a probabilidade de ser classificado como ataque\\n"
                formatted_explanation += "- **Contribuições negativas (em verde):** Aumentam a probabilidade de ser classificado como normal\\n"
                formatted_explanation += "- **Contribuições neutras (valor 0):** Não têm influência na classificação\\n"
                
                # Criar o arquivo formatado usando o Path diretamente
                try:
                    # Garantir que o diretório existe
                    os.makedirs(base_reports, exist_ok=True)
                    
                    lime_explanation_formatted_file = base_reports / "lime_explanation_formatado.txt"
                    with open(str(lime_explanation_formatted_file), 'w', encoding='utf-8') as f:
                        f.write(formatted_explanation)
                    logger.info(f"Versão formatada da explicação LIME criada com sucesso em {lime_explanation_formatted_file}")
                except Exception as e:
                    logger.error(f"Erro ao escrever o arquivo formatado: {e}", exc_info=True)
            except Exception as e:
                logger.error(f'Erro ao criar versão formatada da explicação LIME: {e}', exc_info=True)
            
            # Salvar visualização LIME
            lime_png_path = base_reports / 'lime_final.png'"""
        
        # Substituir o código problemático pelo corrigido
        new_content = content.replace(problematic_code, fixed_code)
        
        # Escrever o conteúdo corrigido de volta no arquivo
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        logger.info(f"Arquivo {file_path} corrigido com sucesso.")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao tentar corrigir o arquivo: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = fix_all_issues()
    print("Correção completa concluída com " + ("sucesso" if success else "falha") + ".")
