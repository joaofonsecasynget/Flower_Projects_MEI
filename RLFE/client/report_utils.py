"""
Utilitários para geração de relatórios e visualizações no cliente federado.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging
import torch
import time
import shap

# Configuração do logger
logger = logging.getLogger()

def generate_evolution_plots(history, base_reports):
    """
    Gera gráficos de evolução das métricas ao longo das rondas.
    
    Args:
        history: Dicionário com o histórico de métricas
        base_reports: Diretório base para salvar os gráficos
        
    Returns:
        list: Lista com os nomes dos arquivos de gráficos gerados
    """
    plot_files = []
    metrics_to_plot = ["train_loss", "val_loss", "test_loss", "val_rmse", "test_rmse"]
    
    # Verificar se há rounds suficientes
    if len(history["train_loss"]) > 0:
        # Criar arrays de rounds para cada métrica (evitando erro de dimensão)
        for metric_key in metrics_to_plot:
            if metric_key in history and len(history[metric_key]) > 0:
                try:
                    plt.figure(figsize=(10, 5))
                    values = history[metric_key]
                    rounds = list(range(1, len(values) + 1))  # Rounds baseados no comprimento real da métrica
                    
                    plt.plot(rounds, values, marker='o', linestyle='-', label=metric_key)
                    plt.xlabel("Ronda")
                    plt.ylabel(metric_key.replace('_', ' ').title())
                    plt.title(f"Evolução do {metric_key.replace('_', ' ').title()}")
                    plt.legend()
                    plt.xticks(rounds)
                    plt.grid(True, axis='y', linestyle='--')

                    plot_filename = f"{metric_key}_evolution.png"
                    plot_path = base_reports / plot_filename
                    plt.savefig(plot_path)
                    plt.close()
                    plot_files.append(plot_filename)
                    logger.info(f"Generated plot: {plot_path}")
                except Exception as e:
                    logger.error(f"Error generating plot for {metric_key}: {e}", exc_info=True)
    else:
        logger.warning("No rounds completed, skipping plot generation.")
        
    return plot_files

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
            .metrics-table th {{ background-color: #1abc9c; color: white; text-transform: capitalize; font-weight: 600; position: sticky; top: 0; z-index: 1; }}
            .metrics-table tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .metrics-table tr:hover {{ background-color: #f1f1f1; }}
            .metrics-table td:first-child, .metrics-table th:first-child {{ text-align: center; font-weight: bold; }}
            .metrics-table td {{ min-width: 80px; }}
            .table-container {{ max-height: 500px; overflow-y: auto; border: 1px solid #ddd; margin-bottom: 20px; }}
            ul {{ list-style-type: square; padding-left: 20px; margin-top: 10px; }}
            li {{ margin-bottom: 5px; }}
            li a {{ color: #3498db; text-decoration: none; }}
            li a:hover {{ text-decoration: underline; }}
            p em {{ color: #7f8c8d; font-size: 0.9em; text-align: center; display: block; margin-top: 20px; }}
            .section {{ background-color: white; padding: 20px; margin-bottom: 20px; border-radius: 5px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }}
            .plot-container img {{ margin-bottom: 25px; }}
            /* Estilos para a tabela de plots */
            .plot-table {{ width: 100%; border-collapse: separate; border-spacing: 15px; }}
            .plot-table td {{ vertical-align: top; text-align: center; width: 33%; background-color: #fff; padding: 10px; border-radius: 4px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); border: 1px solid #eee; }}
            .plot-table img {{ max-width: 100%; height: auto; margin: 10px auto; display: block; }}
            .plot-table h3 {{ margin-top: 0; font-size: 1em; border-bottom: 1px solid #eee; padding-bottom: 5px; margin-bottom: 10px; color: #555; }}
            .explanation-grid {{
                display: grid;
                grid-template-columns: repeat(6, 1fr);
                grid-template-rows: auto auto;
                gap: 20px;
                margin-bottom: 30px;
            }}
            .grid-item {{
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                padding: 15px;
            }}
            .grid-item h4 {{
                margin-top: 0;
                text-align: center;
                color: #2c3e50;
                font-size: 16px;
                border-bottom: 1px solid #eee;
                padding-bottom: 10px;
                margin-bottom: 15px;
            }}
            .grid-item img {{
                display: block;
                margin: 0 auto;
                max-width: 100%;
            }}
            .feature-importance {{
                grid-column: span 3;
            }}
            .temporal-heatmap {{
                grid-column: span 6;
            }}
            .temporal-index {{
                grid-column: span 3;
            }}
            .timestamp-features {{
                grid-column: span 3;
            }}
            .temporal-trend {{
                grid-column: span 6;
            }}
        </style>
    </head>
    <body>
        <h1>Relatório Federado Detalhado - Cliente {client_id}</h1>

        <div class="section">
            <h2>Evolução Detalhada das Métricas e Tempos por Ronda</h2>
    """
    # Gerar Tabela HTML Dinamicamente
    num_rounds_completed = len(history["train_loss"])
    if num_rounds_completed > 0 and history:
        html_content += '<div class="table-container">\n<table class="metrics-table">\n<thead>\n<tr><th>Ronda</th>'
        all_keys = list(history.keys()) # Usar as chaves existentes no histórico
        # Ordenar chaves para melhor leitura (opcional, mas recomendado)
        key_order = ["train_loss", "val_loss", "test_loss", "val_rmse", "test_rmse",
                     "fit_duration_seconds", "evaluate_duration_seconds",
                     "lime_duration_seconds", "shap_duration_seconds"]
        ordered_keys = [k for k in key_order if k in all_keys] + [k for k in all_keys if k not in key_order]

        for key in ordered_keys:
            html_content += f"<th>{key.replace('_', ' ').title()}</th>"
        html_content += "</tr>\n</thead>\n<tbody>\n"

        for r in range(num_rounds_completed):
            html_content += f"<tr><td>{r + 1}</td>"
            for key in ordered_keys: # Iterar com a ordem definida
                value = history.get(key, [None]*num_rounds_completed)[r] # Acesso seguro
                # Formatar: 4 casas decimais para métricas, 4 para tempos. Tratar NaN/None.
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    formatted_value = "N/A"
                elif isinstance(value, (int, float)):
                    if 'duration' in key or 'seconds' in key:
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = f"{value:.4f}"
                else:
                    formatted_value = str(value)
                html_content += f"<td>{formatted_value}</td>"
            html_content += "</tr>\n"
        html_content += "</tbody>\n</table>\n</div>"
    else:
        html_content += "<p><em>Não há dados históricos para apresentar na tabela.</em></p>"
    html_content += "</div>" # Fim da section da tabela


    # Incluir Plots de Evolução Gerados
    html_content += '<div class="section plot-container">\n<h2>Plots de Evolução das Métricas</h2>\n'
    if plot_files:
        html_content += '<table class="plot-table">\n'
        num_plots = len(plot_files)
        plots_per_row = 3
        for i, plot_file in enumerate(plot_files):
            if i % plots_per_row == 0:
                html_content += '<tr>\n'
            
            plot_title = plot_file.replace('_evolution.png','').replace('_', ' ').title()
            html_content += f'<td><h3>{plot_title}</h3><img src="{plot_file}" alt="Evolução {plot_title}" /></td>\n'
            
            if (i + 1) % plots_per_row == 0 or (i + 1) == num_plots:
                # Se for o último plot da linha OU o último plot de todos
                # Adicionar células vazias se a linha não estiver completa
                if (i + 1) == num_plots and (i + 1) % plots_per_row != 0:
                    remaining_cells = plots_per_row - ((i + 1) % plots_per_row)
                    for _ in range(remaining_cells):
                        html_content += '<td></td>\n' # Célula vazia
                html_content += '</tr>\n'
        html_content += '</table>\n'
    else:
         html_content += "<p><em>Nenhum gráfico de evolução foi gerado ou encontrado.</em></p>\n"
    html_content += "</div>" # Fim da section dos plots


    # Secção de Explicabilidade (mantém-se, usa os ficheiros finais gerados antes)
    html_content += f"""
        <div class="section">
            <h2>Explicabilidade (Resultados Finais da Última Ronda)</h2>
            <h3>LIME</h3>
            {f'<img src="lime_final.png" alt="LIME final" style="max-width: 600px;"/>' if 'lime_final.png' in os.listdir(base_reports) else '<p><em>Explicação LIME não disponível ou não gerada.</em></p>'}
            <p><a href="lime_final.html" target="_blank">Ver LIME interativo</a> | <a href="lime_explanation.txt" target="_blank">Ver LIME (texto)</a></p>
            
            <h3>SHAP</h3>
            {f'<img src="shap_final.png" alt="SHAP final" style="max-width: 600px;"/>' if 'shap_final.png' in os.listdir(base_reports) else '<p><em>Explicação SHAP não disponível ou não gerada.</em></p>'}
            <p><a href="shap_values.npy" download>Download SHAP Values (.npy)</a></p>
            
            <h3>Explicabilidade Agregada</h3>
            
            <div class="explanation-grid">
                <!-- Primeiro grupo: Importância por categorias de features -->
                <div class="grid-item feature-importance">
                    <h4>Importância por Tipo de Feature</h4>
                    {f'<img src="shap_feature_type_importance.png" alt="Importância por tipo de feature"/>' if 'shap_feature_type_importance.png' in os.listdir(base_reports) else '<p><em>Gráfico não disponível</em></p>'}
                </div>
                
                <!-- Segundo grupo: Features relacionadas a timestamp -->
                <div class="grid-item timestamp-features">
                    <h4>Importância das Features de Timestamp</h4>
                    {f'<img src="shap_timestamp_features_importance.png" alt="Importância das features de timestamp"/>' if 'shap_timestamp_features_importance.png' in os.listdir(base_reports) else '<p><em>Gráfico não disponível</em></p>'}
                </div>
                
                <!-- Terceiro grupo: Importância do índice temporal -->
                <div class="grid-item temporal-index">
                    <h4>Importância por Índice Temporal</h4>
                    {f'<img src="shap_temporal_index_importance.png" alt="Importância por índice temporal"/>' if 'shap_temporal_index_importance.png' in os.listdir(base_reports) else '<p><em>Gráfico não disponível</em></p>'}
                </div>
                
                <!-- Grupo gráficos largos: Tendência temporal e Heatmap -->
                <div class="grid-item temporal-trend">
                    <h4>Tendências Temporais</h4>
                    {f'<img src="shap_temporal_trends.png" alt="Tendências temporais"/>' if 'shap_temporal_trends.png' in os.listdir(base_reports) else '<p><em>Gráfico não disponível</em></p>'}
                </div>
                
                <div class="grid-item temporal-heatmap">
                    <h4>Mapa de Calor Temporal</h4>
                    {f'<img src="shap_temporal_heatmap.png" alt="Heatmap temporal"/>' if 'shap_temporal_heatmap.png' in os.listdir(base_reports) else '<p><em>Gráfico não disponível</em></p>'}
                </div>
            </div>
        </div>
    """
    # Secção de Artefactos (mantém-se)
    html_content += f"""
        <div class="section">
            <h2>Artefactos e Downloads</h2>
            <ul>
                <li><a href="metrics_history.json" target="_blank">Histórico Completo (Métricas e Tempos - JSON)</a></li>
                <li><a href="../results/client_{client_id}/model_client_{client_id}.pt" download>Modelo Treinado Final (.pt)</a></li>
                <li><a href="info.txt" target="_blank">Informações Gerais (info.txt)</a></li>
                {f'<li><a href="lime_explanation.txt" target="_blank">Explicação LIME (texto)</a></li>' if 'lime_explanation.txt' in os.listdir(base_reports) else ''}
                {f'<li><a href="lime_final.html" target="_blank">Explicação LIME (interativo HTML)</a></li>' if 'lime_final.html' in os.listdir(base_reports) else ''}
                {f'<li><a href="shap_values.npy" download>Valores SHAP (.npy)</a></li>' if 'shap_values.npy' in os.listdir(base_reports) else ''}
            </ul>
        </div>

        <p><em>Relatório gerado automaticamente em {dt_str}</em></p>
    </body>
    </html>
    """
    
    # Salvar o HTML final
    report_path = base_reports / "final_report.html"
    try:
        with open(report_path, "w", encoding='utf-8') as f:
            f.write(html_content)
        logger.info(f"Final HTML report generated at {report_path}")
    except Exception as e:
        logger.error(f"Error saving final HTML report: {e}", exc_info=True)
    
    return report_path

def save_artifacts(base_reports, base_results, model, client_id, dataset_path, seed, epochs, server_address):
    """
    Salva artefatos do modelo e informações gerais.
    
    Args:
        base_reports: Diretório base para relatórios
        base_results: Diretório base para resultados
        model: Modelo treinado
        client_id: ID do cliente
        dataset_path: Caminho para o dataset
        seed: Semente para reprodutibilidade
        epochs: Número de épocas por ronda
        server_address: Endereço do servidor
        
    Returns:
        tuple: Tupla com caminhos para os artefatos salvos (info_path, model_path)
    """
    try:
        # Gerar arquivo de informações gerais
        info_path = base_reports / "info.txt"
        with open(info_path, "w") as f:
            f.write(f"Cliente: {client_id}\n")
            f.write(f"Data: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset Path: {dataset_path}\n")
            f.write(f"Seed: {seed}\n")
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Server Address: {server_address}\n")
        logger.info(f"Info file saved to {info_path}")
        
        # Salvar modelo final
        model_path = base_results / f"model_client_{client_id}.pt"
        try:
            torch.save(model.state_dict(), model_path)
            logger.info(f"Final model saved to {model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            model_path = None
        
        return info_path, model_path
    except Exception as e:
        logger.error(f"Error saving artifacts: {e}", exc_info=True)
        return None, None

def generate_explainability(explainer, X_train, base_reports, sample_idx=0):
    """
    Gera explicabilidade LIME e SHAP para o modelo.
    
    Args:
        explainer: Instância de ModelExplainer
        X_train: Dados de treino
        base_reports: Diretório base para salvar as explicações
        sample_idx: Índice da amostra para explicar com LIME
        
    Returns:
        tuple: Tupla com durações (lime_duration, shap_duration)
    """
    lime_duration = 0.0
    shap_duration = 0.0
    
    try:
        # Carregar X_train se existir (pode ter sido salvo anteriormente)
        if isinstance(X_train, (str, Path)):
            X_train_path = Path(X_train)
            if X_train_path.exists():
                X_train = np.load(X_train_path)
            else:
                logger.error(f"X_train not found at {X_train_path}")
                return lime_duration, shap_duration
        
        # Gerar LIME
        lime_start = time.time()
        try:
            # Escolher instância para explicar (primeira por padrão)
            instance = X_train[sample_idx].reshape(1, -1)[0]
            lime_exp = explainer.explain_lime(X_train, instance)
            
            # Salvar explicação LIME como texto
            lime_txt_path = base_reports / "lime_explanation.txt"
            with open(lime_txt_path, "w") as f:
                f.write(str(lime_exp.as_list()))
            
            # Salvar visualização LIME
            lime_png_path = base_reports / "lime_final.png"
            lime_exp.save_to_file(str(base_reports / "lime_final.html"))
            lime_fig = lime_exp.as_pyplot_figure()
            lime_fig.savefig(lime_png_path)
            plt.close(lime_fig)
            
            logger.info(f"LIME explanation generated and saved to {lime_png_path}")
        except Exception as e:
            logger.error(f"Error generating LIME explanation: {e}", exc_info=True)
        finally:
            lime_duration = time.time() - lime_start
        
        # Gerar SHAP
        shap_start = time.time()
        try:
            # Gerar valores SHAP
            shap_values = explainer.explain_shap(X_train)
            
            # Salvar valores SHAP
            shap_values_path = base_reports / "shap_values.npy"
            np.save(shap_values_path, shap_values)
            
            # Gerar visualização SHAP
            shap_path = base_reports / "shap_final.png"
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_train, feature_names=explainer.feature_names, show=False)
            plt.tight_layout()
            plt.savefig(shap_path)
            plt.close()
            
            # Gerar visualizações de explicabilidade agregada
            try:
                # Visualização de importância por tipo de feature
                type_importance_path = explainer.create_feature_type_importance(base_reports / "shap_feature_type_importance.png")
                logger.info(f"Feature type importance visualization saved to {type_importance_path}")
                
                # Visualização de tendências temporais
                temporal_trend_path = explainer.create_temporal_trend_plot(base_reports / "shap_temporal_trends.png")
                logger.info(f"Temporal trend visualization saved to {temporal_trend_path}")
                
                # Heatmap temporal
                temporal_heatmap_path = explainer.create_temporal_heatmap(base_reports / "shap_temporal_heatmap.png")
                logger.info(f"Temporal heatmap visualization saved to {temporal_heatmap_path}")
                
                # Visualização da importância por número da série temporal
                temporal_index_path = explainer.create_temporal_index_importance(base_reports / "shap_temporal_index_importance.png")
                logger.info(f"Temporal index importance visualization saved to {temporal_index_path}")
                
                # Visualização da importância das features extraídas do timestamp
                timestamp_features_path = explainer.create_timestamp_features_importance(base_reports / "shap_timestamp_features_importance.png") 
                logger.info(f"Timestamp features importance visualization saved to {timestamp_features_path}")
            except Exception as e:
                logger.error(f"Error generating aggregated explainability: {e}", exc_info=True)
            
            logger.info(f"SHAP explanation generated and saved to {shap_path}")
        except Exception as e:
            logger.error(f"Error generating SHAP explanation: {e}", exc_info=True)
        finally:
            shap_duration = time.time() - shap_start
    
    except Exception as e:
        logger.error(f"Error in generate_explainability: {e}", exc_info=True)
    
    return lime_duration, shap_duration
