import os
import json
from datetime import datetime
import logging
import html
import re

# Configuração do logger
logger = logging.getLogger(__name__)

def generate_html_report(history, plot_files, base_reports, client_id, dataset_path, seed, epochs, server_address):
    """
    Gera o relatório HTML final com todas as métricas, gráficos e explicabilidade.
    
    Args:
        history: Dicionário com o histórico de métricas
        plot_files: Lista de arquivos de gráficos gerados
        base_reports: Diretório base para salvar o relatório
        client_id: ID do cliente
        dataset_path: Caminho para o dataset
        seed: Semente para reprodutibilidade
        epochs: Número de épocas por ronda
        server_address: Endereço do servidor
        
    Returns:
        str: Caminho para o relatório HTML gerado
    """
    dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Preparar conteúdo das informações da instância LIME
    lime_instance_html = "<p><em>Informações da instância não disponíveis</em></p>"
    
    # Verificar primeiro o novo arquivo instance_info.json, depois o anterior explained_instance_info.json
    instance_info = None
    
    if 'instance_info.json' in os.listdir(base_reports):
        try:
            # Carregar informações do novo arquivo com todos os valores de features
            with open(base_reports/'instance_info.json', 'r') as f:
                instance_info = json.load(f)
                logger.info(f"Arquivo instance_info.json carregado com sucesso para o relatório")
        except Exception as e:
            logger.error(f"Erro ao carregar instance_info.json: {e}")
    
    # Se não encontrou o novo arquivo, tenta o antigo
    if instance_info is None and 'explained_instance_info.json' in os.listdir(base_reports):
        try:
            # Carregar informações da instância anterior
            with open(base_reports/'explained_instance_info.json', 'r') as f:
                instance_info = json.load(f)
                logger.info(f"Arquivo explained_instance_info.json carregado com sucesso para o relatório")
        except Exception as e:
            logger.error(f"Erro ao carregar explained_instance_info.json: {e}")
    
    # Se qualquer um dos arquivos foi carregado com sucesso    
    if instance_info:
        try:
            # Criar HTML com as informações da instância
            features_rows = ""
            if 'features' in instance_info:
                # Nova estrutura com 'original' e 'normalized'
                for k, v in list(instance_info['features'].items())[:20]:
                    # Verificar se estamos usando a nova estrutura com 'original' e 'normalized'
                    if isinstance(v, dict) and ('original' in v or 'normalized' in v):
                        # Pegar valor original se disponível, senão usar normalizado
                        orig_val = v.get('original', 'N/A')
                        norm_val = v.get('normalized', 'N/A')
                        
                        # Adicionar linha mostrando ambos os valores
                        features_rows += f'<tr><td style="padding: 3px;">{k}</td><td style="text-align: right; padding: 3px;">{orig_val}</td><td style="text-align: right; padding: 3px; color: #777;"><em>{norm_val}</em></td></tr>'
                    else:
                        # Compatibilidade com formato antigo
                        features_rows += f'<tr><td style="padding: 3px;">{k}</td><td style="text-align: right; padding: 3px;">{v}</td><td></td></tr>'
            
            # Verificar se temos informações do target original
            original_target_html = ""
            if instance_info.get('original_target') is not None:
                prediction_result = "CORRETA" if instance_info['prediction_correct'] else "INCORRETA"
                result_color = "green" if instance_info['prediction_correct'] else "red"
                
                original_target_html = f"""
                <p><strong>Valor Real (Target):</strong> {instance_info['original_target_class']} 
                   (valor: {instance_info['original_target']})</p>
                <p><strong>Resultado:</strong> <span style="color: {result_color}; font-weight: bold;">Previsão {prediction_result}</span></p>
                """
            
            lime_instance_html = f"""
            <p><strong>Índice no dataset original:</strong> {instance_info['original_index']}</p>
            <p><strong>Previsão do Modelo:</strong> {instance_info['predicted_class']} 
               (valor: {instance_info['prediction']}, 
               confiança: {instance_info['prediction_confidence']}%)</p>
            {original_target_html}
            <p><strong>Total de Features:</strong> {instance_info['feature_count']}</p>
            <details>
                <summary>Ver valores das features iniciais</summary>
                <div style="max-height: 200px; overflow-y: auto; font-size: 0.85em;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <th style="text-align: left; padding: 3px; border-bottom: 1px solid #ddd;">Feature</th>
                            <th style="text-align: right; padding: 3px; border-bottom: 1px solid #ddd;">Valor Original</th>
                            <th style="text-align: right; padding: 3px; border-bottom: 1px solid #ddd; color: #777;">Valor Normalizado</th>
                        </tr>
                        {features_rows}
                    </table>
                    <p><em>Amostra com as 20 features iniciais com valores originais e normalizados</em></p>
                </div>
            </details>
            """
        except Exception as e:
            logger.error(f"Erro ao processar informações da instância LIME: {e}")
            lime_instance_html = f"<p><em>Erro ao processar informações da instância: {str(e)}</em></p>"

    # --- NOVO: preparar explicação LIME formatada ---
    lime_explanation_path = os.path.join(base_reports, 'lime_explanation_formatado.txt')
    if os.path.exists(lime_explanation_path):
        with open(lime_explanation_path, 'r', encoding='utf-8') as f:
            lime_explanation_text_html = html.escape(f.read())
    else:
        lime_explanation_text_html = 'Arquivo de explicação LIME formatado não encontrado.'
        
    # Verificar existência dos arquivos de imagem e preparar o HTML correspondente
    lime_img_path = os.path.join(base_reports, 'lime_final.png')
    shap_img_path = os.path.join(base_reports, 'shap_final.png')
    
    # HTML para a imagem LIME
    if os.path.exists(lime_img_path):
        lime_img_html = '<img src="lime_final.png" alt="LIME final" style="width:100%; max-width:600px;">'
    else:
        lime_img_html = '<p><em>Visualização LIME não disponível.</em></p>'
    
    # HTML para a imagem SHAP
    if os.path.exists(shap_img_path):
        shap_img_html = '<img src="shap_final.png" alt="SHAP final" style="width:100%; max-width:600px;">'
    else:
        shap_img_html = '<p><em>Visualização SHAP não disponível.</em></p>'

    # --- NOVO: gerar tabelas de métricas por ronda ---
    _time_key_alias = {
        'fit_time': 'fit_duration_seconds',
        'evaluate_time': 'evaluate_duration_seconds',
        'lime_time': 'lime_duration_seconds',
        'shap_time': 'shap_duration_seconds'
    }

    def _get(hist_key, idx, default='N/A', prec=4):
        # aceitar alias
        true_key = hist_key
        if hist_key not in history and hist_key in _time_key_alias:
            true_key = _time_key_alias[hist_key]
        if history and true_key in history and isinstance(history[true_key], list) and idx < len(history[true_key]):
            val = history[true_key][idx]
            # converter string numérica para float
            if isinstance(val, str):
                try:
                    val = float(val)
                except ValueError:
                    return val
            if isinstance(val, float):
                # se é inteiro no valor, mostrar sem decimais
                if val.is_integer():
                    return int(val)
                return round(val, prec)
            if isinstance(val, int):
                return val
            return val
        return default

    def format_value(key, idx, default='N/A', prec=4):
        # Tratamento especial para valores de RMSE
        if key in ['val_rmse', 'test_rmse']:
            val = _get(key, idx, default, prec)
            # Se for string, tentar converter para float
            if isinstance(val, str):
                try:
                    val = float(val)
                except ValueError:
                    return val  # Se falhar, retornar o valor original
            # Agora temos certeza que é um valor numérico
            if isinstance(val, (int, float)):
                return f"{val:.4f}"  # Sempre 4 casas decimais para RMSE
            return val
        else:
            # Para outros valores, usar o comportamento padrão
            val = _get(key, idx, default, prec)
            
            if isinstance(val, (int, float)):
                if isinstance(val, float):
                    # Se é inteiro no valor, mostrar sem decimais
                    if val.is_integer():
                        return str(int(val))
                    # Senão, formatar com precisão específica
                    return f"{val:.{prec}f}"
                return str(val)
            return val

    # Função especial para tempos de LIME e SHAP, que geralmente têm array de tamanho 1
    def _get_explainability_time(key, idx, rounds, default='-', prec=4):
        true_key = _time_key_alias.get(key, key)
        
        if history and true_key in history and isinstance(history[true_key], list):
            values = history[true_key]
            # Lógica para lidar com arrays de métricas de explicabilidade
            
            # Caso 1: Array de tamanho 1 (caso típico LIME/SHAP)
            if len(values) == 1:
                # Mostrar somente na última ronda
                if idx == rounds - 1:
                    val = values[0]
                    if isinstance(val, (int, float)):
                        if isinstance(val, float) and val.is_integer():
                            return str(int(val))
                        return f"{val:.{prec}f}" if isinstance(val, float) else str(val)
                    return val
                else:
                    # Para todas as outras rondas, retornar o valor padrão
                    return default
            
            # Caso 2: Array com múltiplos valores
            elif len(values) > idx:
                # Verificar se o elemento atual do array existe
                val = values[idx]
                if isinstance(val, (int, float)):
                    if isinstance(val, float) and val.is_integer():
                        return str(int(val))
                    return f"{val:.{prec}f}" if isinstance(val, float) else str(val)
                return val
        return default

    train_val_rows, test_rows, timing_rows = "", "", ""
    rounds = len(history.get('train_loss', [])) if history else 0
    for i in range(rounds):
        r = i + 1
        
        train_val_rows += f"<tr><td>{r}</td><td>{format_value('train_loss', i)}</td><td>{format_value('val_loss', i)}</td><td>{format_value('val_rmse', i)}</td><td>{format_value('val_accuracy', i)}</td><td>{format_value('val_precision', i)}</td><td>{format_value('val_recall', i)}</td><td>{format_value('val_f1', i)}</td></tr>"
        test_rows += f"<tr><td>{r}</td><td>{format_value('test_loss', i)}</td><td>{format_value('test_rmse', i)}</td><td>{format_value('test_accuracy', i)}</td><td>{format_value('test_precision', i)}</td><td>{format_value('test_recall', i)}</td><td>{format_value('test_f1', i)}</td></tr>"
        
        # Usar função especial para LIME e SHAP
        lime_time = _get_explainability_time('lime_time', i, rounds, default='-', prec=4)
        shap_time = _get_explainability_time('shap_time', i, rounds, default='-', prec=4)
        timing_rows += f"<tr><td>{r}</td><td>{format_value('fit_time', i, prec=4)}</td><td>{format_value('evaluate_time', i, prec=4)}</td><td>{lime_time}</td><td>{shap_time}</td></tr>"

    # Conteúdo HTML do relatório
    html_content = f"""
    <!DOCTYPE html>
    <html lang="pt">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Relatório Federado Detalhado - Cliente {client_id}</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background-color: #f4f7f6; color: #333; }}
            h1, h2, h3 {{ color: #2c3e50; border-bottom: 2px solid #1abc9c; padding-bottom: 5px;}}
            h1 {{ text-align: center; margin-bottom: 30px; }}
            img {{ max-width: 800px; height: auto; margin: 15px auto; display: block; border: 1px solid #ddd; border-radius: 4px; padding: 5px; background-color: white; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
            .metrics-table {{ border-collapse: collapse; margin: 25px 0; width: 100%; box-shadow: 0 2px 3px rgba(0,0,0,0.1); background-color: white; font-size: 0.9em; }}
            .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px 10px; text-align: left; }}
            .metrics-table th {{ background-color: #1abc9c; color: white; text-transform: capitalize; font-weight:600; position:sticky; top:0; z-index:1; }}
            .metrics-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metrics-table tr:hover {{ background-color:#f1f1f1; }}
            .metrics-table td:first-child, .metrics-table th:first-child {{ text-align:center; font-weight:bold; }}
            .metrics-table td {{ min-width:80px; }}
            .section {{ background-color: white; padding: 20px; margin-bottom: 25px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .grid-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
            .plot-container {{ text-align: center; }}
            .code-block {{ background-color: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 4px; overflow-x: auto; font-family: 'Consolas', 'Monaco', monospace; font-size: 0.85em; }}
            details {{ margin-bottom: 10px; background-color: #eef; padding: 10px; border-radius: 4px; border: 1px solid #dde; }}
            summary {{ font-weight: bold; cursor: pointer; color: #2c3e50; }}
            .footer {{ text-align: center; margin-top: 30px; font-size: 0.8em; color: #777; }}
            /* adições */
            .table-container {{ max-height: 500px; overflow-y:auto; border:1px solid #ddd; margin-bottom:20px; }}
            .evolution-plots {{ display:flex; flex-wrap:wrap; gap:20px; justify-content:center; }}
            .plot-item {{ flex:1 1 400px; max-width:100%; background:#fff; padding:15px; border-radius:4px; box-shadow:0 1px 2px rgba(0,0,0,0.05); border:1px solid #eee; text-align:center; margin-bottom:20px; }}
            .plot-item img {{ max-width:100%; height:auto; margin:10px auto; display:block; }}
            .plot-item h3 {{ margin-top:0; border-bottom:1px solid #eee; padding-bottom:5px; margin-bottom:10px; color:#2c3e50; }}
            /* explanation-grid styles */
            .explanation-grid {{ display:grid; grid-template-columns:repeat(6,1fr); grid-template-rows:auto auto; gap:20px; margin-bottom:30px; }}
            .grid-item {{ background:#fff; border-radius:5px; box-shadow:0 1px 3px rgba(0,0,0,0.1); padding:15px; }}
            .grid-item h4 {{ margin-top:0; text-align:center; color:#2c3e50; font-size:16px; border-bottom:1px solid #eee; padding-bottom:10px; margin-bottom:15px; }}
            .feature-importance {{ grid-column: span 3; }}
            .timestamp-features {{ grid-column: span 3; }}
            .temporal-index {{ grid-column: span 3; }}
            .temporal-trend {{ grid-column: span 6; }}
            .temporal-heatmap {{ grid-column: span 6; }}
            ul {{ list-style-type:square; padding-left:20px; margin-top:10px; }}
            li {{ margin-bottom:5px; }}
            li a {{ color:#3498db; text-decoration:none; }}
            li a:hover {{ text-decoration:underline; }}
        </style>
    </head>
    <body>
        <h1>Relatório Federado Detalhado - Cliente {client_id}</h1>
        
        <div class="section">
            <h2>Evolução Detalhada das Métricas e Tempos por Ronda</h2>
            <h3>Tabela 1: Métricas de Treino e Validação</h3>
            <div class="table-container"><table class="metrics-table"><thead><tr><th>Ronda</th><th>Train Loss</th><th>Val Loss</th><th>Val Rmse</th><th>Val Accuracy</th><th>Val Precision</th><th>Val Recall</th><th>Val F1</th></tr></thead><tbody>{train_val_rows}</tbody></table></div>
            <h3>Tabela 2: Métricas de Teste</h3>
            <div class="table-container"><table class="metrics-table"><thead><tr><th>Ronda</th><th>Test Loss</th><th>Test Rmse</th><th>Test Accuracy</th><th>Test Precision</th><th>Test Recall</th><th>Test F1</th></tr></thead><tbody>{test_rows}</tbody></table></div>
            <h3>Tabela 3: Tempos de Processamento (segundos)</h3>
            <div class="table-container"><table class="metrics-table"><thead><tr><th>Ronda</th><th>Fit</th><th>Evaluate</th><th>Lime</th><th>Shap</th></tr></thead><tbody>{timing_rows}</tbody></table></div>
        </div>

        <div class="section plot-container">
            <h2 style="text-align: left;">Plots de Evolução das Métricas</h2>
            <div class="evolution-plots">
                <div class="plot-item">
                    <h3>Loss Evolution</h3>
                    <img src="losses_evolution.png" alt="Loss Evolution" style="width:100%;" onerror="this.style.display='none'; this.parentElement.innerHTML += '<p><em>Gráfico não disponível</em></p>';">
                </div>
                <div class="plot-item">
                    <h3>RMSE Evolution</h3>
                    <img src="rmse_evolution.png" alt="RMSE Evolution" style="width:100%;" onerror="this.style.display='none'; this.parentElement.innerHTML += '<p><em>Gráfico não disponível</em></p>';">
                </div>
                <div class="plot-item">
                    <h3>Validation Metrics</h3>
                    <img src="validation_metrics_evolution.png" alt="Validation Metrics" style="width:100%;" onerror="this.style.display='none'; this.parentElement.innerHTML += '<p><em>Gráfico não disponível</em></p>';">
                </div>
                <div class="plot-item">
                    <h3>Test Metrics</h3>
                    <img src="test_metrics_evolution.png" alt="Test Metrics" style="width:100%;" onerror="this.style.display='none'; this.parentElement.innerHTML += '<p><em>Gráfico não disponível</em></p>';">
                </div>
                <div class="plot-item">
                    <h3>Processing Times</h3>
                    <img src="processing_times_evolution.png" alt="Processing Times" style="width:100%;" onerror="this.style.display='none'; this.parentElement.innerHTML += '<p><em>Gráfico não disponível</em></p>';">
                </div>
                <div class="plot-item">
                    <h3>Accuracy Comparison</h3>
                    <img src="accuracy_comparison.png" alt="Accuracy Comparison" style="width:100%;" onerror="this.style.display='none'; this.parentElement.innerHTML += '<p><em>Gráfico não disponível</em></p>';">
                </div>
            </div>
        </div>
        
        <div class="section">
            <h2>Explicabilidade (Resultados Finais da Última Ronda)</h2>
            
            <h3>Instância Analisada</h3>
            <div style="background-color:#f8f9fa; padding:15px; border-left:3px solid #1abc9c; margin-bottom:20px;">
                {lime_instance_html}
            </div>
            
            <h3>Tempos de Explicabilidade</h3>
            <img src="explainability_times.png" alt="Tempos de Explicabilidade" style="max-width: 600px;" onerror="this.style.display='none';"/>
            
            <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: space-between; margin-top: 20px;">
                <div style="flex:1; min-width:300px;">
                    <h3>LIME</h3>
                    {lime_img_html}
                    <p><a href="lime_explanation.txt" target="_blank">Ver LIME (texto)</a></p>
                    <p><a href="lime_explanation_formatado.txt" target="_blank">Ver LIME (formatado)</a></p>
                </div>
                <div style="flex:1; min-width:300px;">
                    <h3>SHAP</h3>
                    {shap_img_html}
                    <p><a href="shap_values.npy" download>Download SHAP Values (.npy)</a></p>
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Explicabilidade Agregada</h2>
            <div class="explanation-grid">
                <div class="grid-item feature-importance"><h4>Importância por Tipo de Feature</h4><p><em>Gráfico não disponível</em></p></div>
                <div class="grid-item timestamp-features"><h4>Importância das Features de Timestamp</h4><p><em>Gráfico não disponível</em></p></div>
                <div class="grid-item temporal-index"><h4>Importância por Índice Temporal</h4><p><em>Gráfico não disponível</em></p></div>
                <div class="grid-item temporal-trend"><h4>Tendências Temporais</h4><p><em>Gráfico não disponível</em></p></div>
                <div class="grid-item temporal-heatmap"><h4>Mapa de Calor Temporal</h4><p><em>Gráfico não disponível</em></p></div>
            </div>
        </div>

        <div class="section">
            <h2>Artefactos e Downloads</h2>
            <ul>
                <li><a href="metrics_history.json" target="_blank">Histórico Completo (JSON)</a></li>
                <li><a href="info.txt" target="_blank">Informações Gerais (info.txt)</a></li>
                <li><a href="lime_explanation_formatado.txt" target="_blank">Explicação LIME (formatado)</a></li>
                <li><a href="shap_values.npy" download>Valores SHAP (.npy)</a></li>
            </ul>
        </div>
        
        <div class="footer">
            <p><em>Relatório gerado automaticamente em {dt_str}</em></p>
        </div>
    </body>
    </html>
    """
    
    report_path = base_reports / f"detailed_report_client_{client_id}.html"
    
    # Armazenar o HTML original para caso a modificação falhe
    original_html = html_content

    try:
        # Encontrar padrões específicos de RMSE e formatá-los
        # Padrão mais específico para valores de RMSE na tabela
        pattern = r'<td>(Val RMSE|Test RMSE)</td>\s*<td>(0\.\d+)</td>'
        
        # Função para formatar o valor
        def format_rmse(match):
            try:
                metric_name = match.group(1)
                val = float(match.group(2))
                return f'<td>{metric_name}</td><td>{val:.4f}</td>'
            except:
                # Se a conversão falhar, manter o original
                return match.group(0)
        
        # Aplicar a substituição
        html_content = re.sub(pattern, format_rmse, html_content)
        
        logger.info("Pós-processamento do HTML para formatação de RMSE realizado com sucesso")
    except Exception as e:
        # Se algo der errado, usar o HTML original
        html_content = original_html
        logger.error(f"Erro no pós-processamento do HTML: {e}")
        
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Relatório HTML gerado: {report_path}")
    return str(report_path)
