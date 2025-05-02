"""
Funções para geração de relatórios HTML com explicações de modelos.
"""
import logging
import pandas as pd
import numpy as np
import os

from .feature_utils import get_feature_category

# Configurar logger
logger = logging.getLogger("explainability.html_report")

def valor_da_feature(feature, feature_names, instance_values):
    """Retorna o valor real da feature na instância"""
    try:
        idx = feature_names.index(feature)
        return instance_values[idx]
    except (ValueError, IndexError):
        return "N/A"

def gerar_tabela_shap(shap_values, feature_names, instance_values, category_values):
    """
    Gera uma tabela HTML com os valores SHAP.
    
    Args:
        shap_values: Valores SHAP para a instância
        feature_names: Lista de nomes das features
        instance_values: Valores originais das features (sem normalização)
        category_values: Dicionário com valores agrupados por categoria
    
    Returns:
        String com HTML da tabela SHAP
    """
    # Criar dicionário para facilitar ordenação e processamento
    shap_data = []
    for i, (name, shap_value) in enumerate(zip(feature_names, shap_values)):
        category = get_feature_category(name)
        # Usar valores originais das features
        value = instance_values[i] if i < len(instance_values) else 0.0
        
        shap_data.append({
            "index": i,
            "feature": name,
            "value": value,
            "shap_value": shap_value,
            "abs_shap_value": abs(shap_value),
            "category": category
        })
    
    # Ordenar por magnitude do valor SHAP
    shap_data = sorted(shap_data, key=lambda x: x["abs_shap_value"], reverse=True)
    
    # Gerar tabela HTML
    html = """
    <div class="shap-table">
        <h3>Top 20 Features com Maior Impacto (SHAP)</h3>
        <table class="table table-striped table-sm">
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Categoria</th>
                    <th>Valor na Instância</th>
                    <th>Valor SHAP</th>
                    <th>Direção</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Incluir as top 20 features
    for i, item in enumerate(shap_data[:20]):
        feature = item["feature"]
        category = item["category"]
        value = item["value"]
        shap_value = item["shap_value"]
        
        # Determinar direção da influência
        if shap_value > 0:
            direction = "Aumenta"
            color = "text-success"
            arrow = "↑"
        else:
            direction = "Diminui"
            color = "text-danger"
            arrow = "↓"
        
        # Formatar valores numéricos
        if value == int(value):
            formatted_value = f"{int(value)}"
        else:
            formatted_value = f"{value:.4f}" if abs(value) <= 1000 else f"{value:.1f}"
        
        html += f"""
            <tr>
                <td>{feature}</td>
                <td>{category}</td>
                <td>{formatted_value}</td>
                <td>{abs(shap_value):.6f}</td>
                <td class="{color}">{arrow} {direction}</td>
            </tr>
        """
    
    html += """
            </tbody>
        </table>
    </div>
    """
    
    return html

def generate_html_report(
    output_dir, 
    prediction, 
    class_label, 
    client_id, 
    instance_idx_desc, 
    instance_original_index, 
    lime_exp, 
    feature_names, 
    instance_values, 
    shap_values, 
    expected_value=None, 
    shap_table_html=None, 
    category_values=None, 
    dataset_origin=None,
    target_value=None
):
    """
    Gera um relatório HTML completo para a explicação da instância.
    
    Args:
        output_dir: Diretório onde o relatório será salvo
        prediction: Valor da previsão do modelo
        class_label: Rótulo da classe prevista
        client_id: ID do cliente
        instance_idx_desc: Descrição do índice da instância
        instance_original_index: Índice original da instância no dataset completo
        lime_exp: Objeto com a explicação LIME
        feature_names: Lista de nomes das features
        instance_values: Lista de valores das features
        shap_values: Lista de valores SHAP
        expected_value: Valor esperado do modelo SHAP
        shap_table_html: HTML da tabela SHAP (opcional)
        category_values: Dicionário com valores agrupados por categoria
        dataset_origin: String indicando a origem dos dados
        target_value: Valor real do target para a instância (opcional)
    """
    logger.info("Gerando relatório HTML consolidado para a instância")
    
    # Obter diretório do script atual para acessar recursos
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Carregar template HTML
    template_path = os.path.join(current_dir, 'templates', 'report_template.html')
    if not os.path.exists(template_path):
        logger.error(f"Template não encontrado em {template_path}")
        template = """
        <html><body>
        <h1>{{ TITLE }}</h1>
        <p>Erro: Template não encontrado. Este é um relatório mínimo.</p>
        <p>Previsão: {{ PREDICTION_VALUE }} ({{ PREDICTION_CLASS }})</p>
        </body></html>
        """
    else:
        with open(template_path, 'r') as file:
            template = file.read()
    
    # Gerar plots e salvar como imagens
    lime_img_path = os.path.join(output_dir, "lime_explanation.png")
    
    # Preparar as seções do relatório
    index_info = f"{instance_idx_desc}"
    # Só mostrar o índice original se for diferente do índice da instância
    if instance_original_index is not None and str(instance_original_index) not in instance_idx_desc:
        index_info += f" (índice original: {instance_original_index})"
    
    # Informação adicional sobre a origem dos dados
    dataset_origin_info = f"{dataset_origin}" if dataset_origin else ""
    
    # Preparar informações sobre o valor real do target e comparação com a previsão
    target_info = ""
    if target_value is not None:
        # Verificar se a previsão coincide com o valor real do target
        if target_value == 0 and float(prediction) < 0:  # Valor negativo representa 'Normal'
            acerto = "Correto"
            accuracy_class = "text-success"
        elif target_value == 1 and float(prediction) >= 0:  # Valor positivo representa 'Ataque'
            acerto = "Correto"
            accuracy_class = "text-success"
        else:
            acerto = "Incorreto"
            accuracy_class = "text-danger"
            
        # Formatar o valor real do target para exibição
        target_label = "Normal" if target_value == 0 else "Ataque"
        target_class = "normal" if target_value == 0 else "attack"
        
        # Construir o HTML com as cores apropriadas
        target_info = '<p><strong>Valor real:</strong> <span class="'
        target_info += target_class
        target_info += '">'
        target_info += target_label
        target_info += '</span></p>'
        
        accuracy_class = "success" if acerto == "Correto" else "danger"
        
        target_info += '<p><strong>Comparação:</strong> <span class="'
        target_info += accuracy_class
        target_info += '">'
        target_info += acerto
        target_info += '</span></p>'
    
    # Gerar tabela para mostrar os valores principais da instância
    instance_values_html = "<h3>Valores Iniciais da Instância</h3>"
    instance_values_html += "<p>Esta tabela mostra alguns dos valores iniciais da instância. "
    instance_values_html += "<a href='instance_values_full.txt' target='_blank'>Ver todos os valores da instância</a></p>"
    instance_values_html += "<table class='table table-striped table-sm'>"
    instance_values_html += "<thead><tr><th>Feature</th><th>Valor</th></tr></thead><tbody>"
    
    # Incluir dl_bitrate_1 e outras features importantes mesmo que não tenham impacto SHAP
    # Removidas "imeisv" e "indice" pois são irrelevantes para o treinamento
    important_features = ["dl_bitrate_1", "ul_bitrate_1"]
    
    # Adicionar as features importantes
    for feature in important_features:
        if feature in feature_names:
            idx = feature_names.index(feature)
            value = instance_values[idx]
            # Formatar valores numéricos grandes
            if value == int(value):
                formatted_value = f"{int(value)}"
            else:
                formatted_value = f"{value:.4f}" if abs(value) <= 1000 else f"{value:.1f}"
            # Remover o tag <strong> para evitar confusão com o cabeçalho
            instance_values_html += f"<tr><td>{feature}</td><td>{formatted_value}</td></tr>"
    
    # Adicionar outras features se necessário para completar 10 rows
    features_added = len(important_features)
    for i, (name, value) in enumerate(zip(feature_names, instance_values)):
        if name not in important_features and features_added < 10:
            if "extra_feature" not in name and "indice" not in name.lower():  # Ignorar features extras e índices
                # Formatar valores numéricos grandes
                if value == int(value):
                    formatted_value = f"{int(value)}"
                else:
                    formatted_value = f"{value:.4f}" if abs(value) <= 1000 else f"{value:.1f}"
                instance_values_html += f"<tr><td>{name}</td><td>{formatted_value}</td></tr>"
                features_added += 1
    
    instance_values_html += "</tbody></table>"
    
    # Gerar tabela HTML para o LIME
    lime_table_html = "<table class='table table-striped table-sm'>"
    lime_table_html += "<thead><tr><th>Feature</th><th>Valor</th><th>Impacto</th></tr></thead><tbody>"
    
    lime_features = lime_exp.as_list()
    for feature, impact in lime_features[:10]:  # Top 10 features
        color = "text-success" if impact > 0 else "text-danger"
        direction = "Positivo" if impact > 0 else "Negativo"
        
        # Encontrar o valor da feature
        value = "N/A"
        if feature in feature_names:
            idx = feature_names.index(feature)
            if idx < len(instance_values):
                # Formatar valores numéricos grandes
                value = instance_values[idx]
                if value == int(value):
                    formatted_value = f"{int(value)}"
                else:
                    formatted_value = f"{value:.4f}" if abs(value) <= 1000 else f"{value:.1f}"
                value = formatted_value
        
        lime_table_html += f"<tr><td>{feature}</td><td>{value}</td><td class='{color}'>{direction} ({impact:.4f})</td></tr>"
    
    lime_table_html += "</tbody></table>"

    # Gerar seção de features por categoria
    categories_html = ""
    if category_values:
        for category, values in category_values.items():
            if len(values) == 0:
                continue
                
            # Ordenar por impacto absoluto
            sorted_values = sorted(values, key=lambda x: abs(x['shap_value']), reverse=True)
            
            categories_html += f"<div class='category-section mb-4'>"
            categories_html += f"<h4>{category.replace('_', ' ').title()}</h4>"
            categories_html += "<table class='table table-sm'>"
            categories_html += "<thead><tr><th>Feature</th><th>Valor</th><th>Impacto SHAP</th></tr></thead><tbody>"
            
            # Mostrar apenas as top 5 features por categoria
            for item in sorted_values[:5]:
                feature = item['feature']
                value = item['value']  # Usar o valor original
                shap_value = item['shap_value']
                
                color = "text-success" if shap_value > 0 else "text-danger"
                direction = "↑" if shap_value > 0 else "↓"
                
                if value == int(value):
                    formatted_value = f"{int(value)}"
                else:
                    formatted_value = f"{value:.4f}" if abs(value) <= 1000 else f"{value:.1f}"
                
                categories_html += f"<tr><td>{feature}</td><td>{formatted_value}</td><td class='{color}'>{direction} {abs(shap_value):.6f}</td></tr>"
            
            categories_html += "</tbody></table></div>"
    
    # Gerar tabela SHAP
    if shap_table_html is None:
        shap_table_html = gerar_tabela_shap(shap_values, feature_names, instance_values, category_values)
    
    # Componentes para substituir no template
    components = {
        'TITLE': 'Relatório de Explicabilidade - RLFE',
        'PREDICTION_VALUE': f"{prediction:.0f} ({class_label})",
        'PREDICTION_CLASS': 'normal' if 'Normal' in class_label else 'attack',
        'CLIENT_ID': str(client_id),
        'INSTANCE_INDEX': index_info,
        'DATASET_ORIGIN': dataset_origin_info,
        'INSTANCE_VALUES': instance_values_html,  # Nova seção com valores da instância
        'LIME_EXPLANATION': lime_table_html,
        'LIME_IMAGE': "lime_explanation.png",
        'SHAP_TABLE': shap_table_html if shap_table_html else "",
        'CATEGORIES_SECTION': categories_html if categories_html else "",
        'TARGET_INFO': target_info if target_info else ""
    }
    
    # Substituir componentes no template
    html_content = template
    for key, value in components.items():
        html_content = html_content.replace(f"{{{{ {key} }}}}", value)
    
    # Salvar o relatório HTML
    report_path = os.path.join(output_dir, "instance_report.html")
    with open(report_path, 'w') as file:
        file.write(html_content)
    
    return report_path
