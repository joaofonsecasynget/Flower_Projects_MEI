"""
Funções para visualização e formatação de explicações SHAP.
"""
import logging
import numpy as np
import pandas as pd

from .feature_utils import get_feature_category

# Configurar logger
logger = logging.getLogger("explainability.shap_visualizations")

def gerar_tabela_shap(shap_values, feature_names, instance, category_values=None):
    """
    Gera uma tabela HTML formatada com as top features por importância SHAP.
    A tabela é colorida por categoria de feature e mostra valores positivos em verde,
    negativos em vermelho.
    
    Args:
        shap_values: Valores SHAP para a instância
        feature_names: Lista de nomes das features
        instance: Valores das features na instância
        category_values: Dicionário de valores agrupados por categoria
    """
    # Criar um DataFrame com os valores SHAP
    data = {
        'Feature': [],
        'SHAP Value': [],
        'Feature Value': [],
        'Category': []
    }
    
    # Garantir que os arrays tenham mesmo comprimento
    min_length = min(len(feature_names), len(shap_values), len(instance))
    
    # Preencher o DataFrame
    for i in range(min_length):
        data['Feature'].append(feature_names[i])
        data['SHAP Value'].append(shap_values[i])
        data['Feature Value'].append(instance[i])
        data['Category'].append(get_feature_category(feature_names[i]))
    
    df = pd.DataFrame(data)
    
    # Ordenar por valor absoluto de SHAP
    df['Absolute'] = df['SHAP Value'].abs()
    df = df.sort_values('Absolute', ascending=False).head(15).drop('Absolute', axis=1)
    
    # Definir cores por categoria
    category_colors = {
        'dl_bitrate': '#1f77b4',   # azul
        'ul_bitrate': '#ff7f0e',   # laranja
        'cell_x_dl_retx': '#2ca02c',  # verde
        'cell_x_dl_tx': '#d62728',  # vermelho
        'cell_x_ul_retx': '#9467bd',  # roxo
        'cell_x_ul_tx': '#8c564b',  # marrom
        'ul_total_bytes_non_incr': '#e377c2',  # rosa
        'dl_total_bytes_non_incr': '#7f7f7f',  # cinza
        'timestamp': '#bcbd22',    # amarelo esverdeado
        'other': '#17becf'         # azul claro
    }
    
    # HTML para a tabela SHAP
    html = '<div class="shap-table">\n'
    html += '<h3>Top Features por Importância SHAP</h3>\n'
    html += '<table class="table table-striped table-sm">\n'
    html += '  <tr><th>Feature</th><th>Categoria</th><th>Valor da Feature</th><th>Contribuição SHAP</th></tr>\n'
    
    for idx, row in df.iterrows():
        feature = row['Feature']
        category = row['Category']
        feature_val = row['Feature Value']
        shap_val = row['SHAP Value']
        
        # Determinar a cor de fundo para a categoria
        bg_color = category_colors.get(category, '#17becf')  # cor padrão se categoria não for encontrada
        
        # Determinar a cor do texto para o valor SHAP
        if shap_val > 0:
            shap_color = 'text-success'
            prefix = '+'
        else:
            shap_color = 'text-danger'
            prefix = ''
        
        # Formatar valores numéricos
        formatted_val = f"{feature_val:.4f}" if abs(feature_val) < 10000 else f"{feature_val:.1f}"
        
        html += f'  <tr>\n'
        html += f'    <td>{feature}</td>\n'
        html += f'    <td>{category}</td>\n'
        html += f'    <td>{formatted_val}</td>\n'
        html += f'    <td class="{shap_color}">{prefix}{shap_val:.6f}</td>\n'
        html += f'  </tr>\n'
    
    html += '</table>\n'
    
    # Adicionar tabelas por categoria
    if category_values:
        html += '<h3>Principais Features por Categoria</h3>\n'
        
        for category, values in category_values.items():
            # Verificar se a lista de valores tem elementos
            if isinstance(values, list) and len(values) > 0:
                bg_color = category_colors.get(category, '#17becf')
                html += f'<div class="category-section mb-4">\n'
                html += f'<h4>{category.replace("_", " ").title()}</h4>\n'
                html += '<table class="table table-sm">\n'
                html += '  <tr><th>Feature</th><th>Valor</th><th>Contribuição SHAP</th></tr>\n'
                
                # Ordenar por magnitude do valor SHAP
                sorted_values = sorted(values, key=lambda x: abs(x.get('shap_value', 0)), reverse=True)
                
                # Mostrar apenas os top 5 por categoria
                for item in sorted_values[:5]:
                    feature = item.get('feature', '')
                    value = item.get('value', 0)
                    shap_value = item.get('shap_value', 0)
                    
                    # Determinar direção e cor
                    if shap_value > 0:
                        direction = "↑"
                        color = "text-success"
                    else:
                        direction = "↓"
                        color = "text-danger"
                        
                    # Formatar valores
                    formatted_val = f"{value:.4f}" if abs(value) < 10000 else f"{value:.1f}"
                    
                    html += f'  <tr>\n'
                    html += f'    <td>{feature}</td>\n'
                    html += f'    <td>{formatted_val}</td>\n'
                    html += f'    <td class="{color}">{direction} {abs(shap_value):.6f}</td>\n'
                    html += f'  </tr>\n'
                
                html += '</table>\n'
                html += '</div>\n'
            # Suporte ao formato antigo (DataFrame)
            elif hasattr(values, 'empty') and not values.empty:
                bg_color = category_colors.get(category, '#17becf')
                html += f'<div class="category-section">\n'
                html += f'<h4 style="background-color: {bg_color};">{category.upper()}</h4>\n'
                html += '<table class="category-table">\n'
                html += '  <tr><th>Feature</th><th>Valor</th><th>Contribuição SHAP</th></tr>\n'
                
                for idx, row in values.iterrows():
                    feature = row['Feature']
                    value = row['Feature Value']
                    shap_value = row['SHAP Value']
                    
                    # Determinar direção e cor
                    if shap_value > 0:
                        direction = "Aumenta"
                        color = "positive"
                    else:
                        direction = "Diminui"
                        color = "negative"
                    
                    html += f'  <tr>\n'
                    html += f'    <td>{feature}</td>\n'
                    html += f'    <td>{value:.4f}</td>\n'
                    html += f'    <td class="{color}">{direction} ({abs(shap_value):.6f})</td>\n'
                    html += f'  </tr>\n'
                
                html += '</table>\n'
                html += '</div>\n'
    
    html += '</div>\n'
    
    return html
