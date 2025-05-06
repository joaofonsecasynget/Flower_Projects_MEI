import os
import json
from datetime import datetime
import logging

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
    if 'lime_instance_info.json' in os.listdir(base_reports):
        try:
            # Carregar informações da instância
            with open(base_reports/'lime_instance_info.json', 'r') as f:
                instance_info = json.load(f)
            
            # Criar HTML com as informações da instância
            features_rows = ""
            if 'features' in instance_info:
                for k, v in list(instance_info['features'].items())[:20]:
                    features_rows += f'<tr><td style="padding: 3px;">{k}</td><td style="text-align: right; padding: 3px;">{v}</td></tr>'
            
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
                <summary>Ver valores das features principais</summary>
                <div style="max-height: 200px; overflow-y: auto; font-size: 0.85em;">
                    <table style="width: 100%; border-collapse: collapse;">
                        <tr>
                            <th style="text-align: left; padding: 3px; border-bottom: 1px solid #ddd;">Feature</th>
                            <th style="text-align: right; padding: 3px; border-bottom: 1px solid #ddd;">Valor</th>
                        </tr>
                        {features_rows}
                    </table>
                    <p><em>Mostrando até 20 features não-zero com valores originais (não normalizados)</em></p>
                </div>
            </details>
            """
        except Exception as e:
            logger.error(f"Erro ao processar informações da instância LIME: {e}")
            lime_instance_html = f"<p><em>Erro ao processar informações da instância: {str(e)}</em></p>"
    
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
            .metrics-table th {{ background-color: #1abc9c; color: white; text-transform: capitalize; }}
            .metrics-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .section {{ background-color: white; padding: 20px; margin-bottom: 25px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .grid-container {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr)); gap: 20px; }}
            .plot-container {{ text-align: center; }}
            .code-block {{ background-color: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 4px; overflow-x: auto; font-family: 'Consolas', 'Monaco', monospace; font-size: 0.85em; }}
            details {{ margin-bottom: 10px; background-color: #eef; padding: 10px; border-radius: 4px; border: 1px solid #dde; }}
            summary {{ font-weight: bold; cursor: pointer; color: #2c3e50; }}
            .footer {{ text-align: center; margin-top: 30px; font-size: 0.8em; color: #777; }}
        </style>
    </head>
    <body>
        <h1>Relatório Detalhado do Cliente {client_id}</h1>
        <div class="section">
            <h2>Informações Gerais</h2>
            <p><strong>ID do Cliente:</strong> {client_id}</p>
            <p><strong>Data de Geração:</strong> {dt_str}</p>
            <p><strong>Dataset Utilizado:</strong> {dataset_path}</p>
            <p><strong>Semente Aleatória:</strong> {seed}</p>
            <p><strong>Épocas por Ronda:</strong> {epochs}</p>
            <p><strong>Endereço do Servidor:</strong> {server_address}</p>
        </div>

        <div class="section">
            <h2>Métricas de Avaliação Final</h2>
            <table class="metrics-table">
                <tr><th>Métrica</th><th>Valor</th></tr>
                {''.join([f"<tr><td>{k.replace('_', ' ').title()}</td><td>{v if isinstance(v, str) else round(v, 4) if isinstance(v, float) else v}</td></tr>" for k, v in history.get('final_metrics', {}).items()])}
            </table>
        </div>

        <div class="section">
            <h2>Evolução das Métricas</h2>
            <div class="grid-container">
                {''.join([f'<div class="plot-container"><h3>{plot_file.replace("_", " ").replace(".png", "").title()}</h3><img src="{plot_file}" alt="{plot_file}"></div>' for plot_file in plot_files if "time" not in plot_file.lower()])}
            </div>
        </div>
        
        <div class="section">
            <h2>Tempos de Processamento</h2>
            <div class="grid-container">
                 {''.join([f'<div class="plot-container"><h3>{plot_file.replace("_", " ").replace(".png", "").title()}</h3><img src="{plot_file}" alt="{plot_file}"></div>' for plot_file in plot_files if "time" in plot_file.lower() and "explainability" not in plot_file.lower()])}
                 <div class="plot-container">
                    <h3>Tempos de Explicabilidade</h3>
                    <img src="explainability_times.png" alt="Tempos de Explicabilidade" onerror="this.style.display='none'; this.parentElement.innerHTML += '<p><em>Gráfico de tempos de explicabilidade não disponível.</em></p>';">
                </div>
            </div>
        </div>

        <div class="section">
            <h2>Explicabilidade LIME</h2>
            <div class="plot-container">
                <h3>Visualização LIME</h3>
                <img src="lime_final.png" alt="Visualização LIME" onerror="this.style.display='none'; this.parentElement.innerHTML += '<p><em>Visualização LIME não disponível.</em></p>';">
            </div>
            <div class="lime-instance-info">
                <h3>Informações da Instância Explicada (LIME)</h3>
                {lime_instance_html}
            </div>
            
            <h3>Explicação LIME Formatada (Texto)</h3>
            <details>
                <summary>Ver explicação LIME formatada completa</summary>
                <div class="code-block">
                    <pre>{ open(base_reports / 'lime_explanation_formatado.txt', 'r').read() if os.path.exists(base_reports / 'lime_explanation_formatado.txt') else 'Arquivo de explicação LIME formatado não encontrado.' }</pre>
                </div>
            </details>
        </div>
        
        <div class="section">
            <h2>Explicabilidade SHAP</h2>
            <div class="plot-container">
                <h3>Visualização SHAP</h3>
                <img src="shap_final.png" alt="Visualização SHAP" onerror="this.style.display='none'; this.parentElement.innerHTML += '<p><em>Visualização SHAP não disponível.</em></p>';">
            </div>
             <h3>Valores SHAP (Dados Brutos)</h3>
            <details>
                <summary>Informações sobre os valores SHAP</summary>
                <p>Os valores SHAP brutos foram salvos em <code>shap_values.npy</code> no diretório de relatórios.</p>
                <p>Estes valores podem ser carregados usando NumPy (<code>np.load('shap_values.npy', allow_pickle=True)</code>) para análises mais aprofundadas.</p>
            </details>
        </div>

        <div class="footer">
            <p>Relatório gerado pela plataforma de Aprendizagem Federada.</p>
        </div>
    </body>
    </html>
    """
    
    report_path = base_reports / f"detailed_report_client_{client_id}.html"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    logger.info(f"Relatório HTML gerado: {report_path}")
    return str(report_path)
