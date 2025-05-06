"""
Utilitários para geração de relatórios e visualizações no cliente federado.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import torch
import time
import shap
import re
from datetime import datetime

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
    
    # Verificar se há rounds suficientes
    if len(history.get("train_loss", [])) <= 1:
        logger.warning("Menos de 2 rounds completos, gráficos de evolução não seriam significativos.")
        return plot_files
    
    # Definir estilo global dos gráficos
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Configurar paletas de cores para consistência entre gráficos
    colors = {
        'train': '#1f77b4',  # azul
        'val': '#ff7f0e',    # laranja
        'test': '#2ca02c',   # verde
        
        'accuracy': '#9467bd',      # roxo
        'precision': '#d62728',     # vermelho
        'recall': '#8c564b',        # marrom
        'f1': '#e377c2',            # rosa
        
        'fit': '#bcbd22',           # amarelo-oliva
        'evaluate': '#17becf',      # ciano
        'lime': '#1a55a3',          # azul escuro
        'shap': '#ff9896'           # salmão
    }
    
    # Criar diretório para gráficos se não existir
    if not base_reports.exists():
        base_reports.mkdir(parents=True)
    
    try:
        # 1. GRÁFICO DE PERDAS (LOSS) - Comparativo treino, validação, teste
        if all(key in history for key in ['train_loss', 'val_loss']):
            plt.figure(figsize=(10, 6))
            
            # Eixo X compartilhado para todas as séries
            rounds = list(range(1, len(history['train_loss']) + 1))
            
            # Plotar cada série
            plt.plot(rounds, history['train_loss'], marker='o', color=colors['train'], 
                    linestyle='-', linewidth=2, label='Treino')
            
            plt.plot(rounds, history['val_loss'], marker='s', color=colors['val'], 
                    linestyle='-', linewidth=2, label='Validação')
            
            if 'test_loss' in history:
                plt.plot(rounds, history['test_loss'], marker='^', color=colors['test'], 
                        linestyle='-', linewidth=2, label='Teste')
            
            # Configurar gráfico
            plt.xlabel("Ronda", fontsize=12, fontweight='bold')
            plt.ylabel("Loss (BCE)", fontsize=12, fontweight='bold')
            plt.title("Evolução da Perda por Ronda", fontsize=14, fontweight='bold')
            plt.legend(fontsize=12, framealpha=0.9)
            plt.xticks(rounds, fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            # Adicionar limites dinâmicos no eixo Y com um pouco de margem
            all_losses = (
                history['train_loss'] + 
                history['val_loss'] + 
                history.get('test_loss', [])
            )
            min_loss = min(all_losses) * 0.9
            max_loss = max(all_losses) * 1.1
            plt.ylim(min_loss, max_loss)
            
            plt.tight_layout()
            plot_filename = "losses_evolution.png"
            plot_path = base_reports / plot_filename
            plt.savefig(plot_path, dpi=120)
            plt.close()
            plot_files.append(plot_filename)
            logger.info(f"Generated plot: {plot_path}")
        
        # 2. GRÁFICO DE RMSE - Comparativo validação e teste
        if 'val_rmse' in history:
            plt.figure(figsize=(10, 6))
            
            rounds = list(range(1, len(history['val_rmse']) + 1))
            
            plt.plot(rounds, history['val_rmse'], marker='s', color=colors['val'], 
                   linestyle='-', linewidth=2, label='Validação')
            
            if 'test_rmse' in history:
                plt.plot(rounds, history['test_rmse'], marker='^', color=colors['test'], 
                       linestyle='-', linewidth=2, label='Teste')
            
            plt.xlabel("Ronda", fontsize=12, fontweight='bold')
            plt.ylabel("RMSE", fontsize=12, fontweight='bold')
            plt.title("Evolução do RMSE por Ronda", fontsize=14, fontweight='bold')
            plt.legend(fontsize=12, framealpha=0.9)
            plt.xticks(rounds, fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plot_filename = "rmse_evolution.png"
            plot_path = base_reports / plot_filename
            plt.savefig(plot_path, dpi=120)
            plt.close()
            plot_files.append(plot_filename)
            logger.info(f"Generated plot: {plot_path}")
        
        # 3. GRÁFICO DE MÉTRICAS DE CLASSIFICAÇÃO (ACC, PREC, REC, F1) - Validação
        metrics_keys = ['val_accuracy', 'val_precision', 'val_recall', 'val_f1']
        if any(key in history for key in metrics_keys):
            plt.figure(figsize=(10, 6))
            
            available_metrics = [m for m in metrics_keys if m in history and len(history[m]) > 0]
            
            max_len = max([len(history[m]) for m in available_metrics]) if available_metrics else 0
            rounds = list(range(1, max_len + 1))
            
            for metric in available_metrics:
                # Extrair o nome da métrica sem o prefixo 'val_'
                metric_name = metric.replace('val_', '')
                metric_color = colors.get(metric_name, '#333333')
                
                values = history[metric]
                if len(values) < max_len:
                    values = values + [None] * (max_len - len(values))
                
                valid_points = [(r, v) for r, v in zip(rounds, values) if v is not None]
                if valid_points:
                    valid_rounds, valid_values = zip(*valid_points)
                    plt.plot(valid_rounds, valid_values, marker='o', color=metric_color, 
                           linestyle='-', linewidth=2, label=metric_name.title())
            
            plt.xlabel("Ronda", fontsize=12, fontweight='bold')
            plt.ylabel("Valor", fontsize=12, fontweight='bold')
            plt.title("Evolução das Métricas de Classificação (Validação)", fontsize=14, fontweight='bold')
            
            # Configurar o intervalo do eixo Y de 0.5 a 1.0 para destacar as diferenças
            # Ou fazer zoom em torno dos valores reais se forem todos muito altos
            min_val = min([min(history[m]) for m in available_metrics if m in history and len(history[m]) > 0])
            min_y = min(0.5, min_val * 0.95) if min_val < 0.9 else 0.9
            plt.ylim(min_y, 1.01)  # Ligeiramente acima de 1 para mostrar pontos no valor 1
            
            plt.legend(fontsize=12, framealpha=0.9)
            plt.xticks(rounds, fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plot_filename = "validation_metrics_evolution.png"
            plot_path = base_reports / plot_filename
            plt.savefig(plot_path, dpi=120)
            plt.close()
            plot_files.append(plot_filename)
            logger.info(f"Generated plot: {plot_path}")
            
        # 4. GRÁFICO DE MÉTRICAS DE CLASSIFICAÇÃO (ACC, PREC, REC, F1) - Teste
        metrics_keys = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        if any(key in history for key in metrics_keys):
            plt.figure(figsize=(10, 6))
            
            available_metrics = [m for m in metrics_keys if m in history and len(history[m]) > 0]
            
            max_len = max([len(history[m]) for m in available_metrics]) if available_metrics else 0
            rounds = list(range(1, max_len + 1))
            
            for metric in available_metrics:
                # Extrair o nome da métrica sem o prefixo 'test_'
                metric_name = metric.replace('test_', '')
                metric_color = colors.get(metric_name, '#333333')
                
                values = history[metric]
                if len(values) < max_len:
                    values = values + [None] * (max_len - len(values))
                
                valid_points = [(r, v) for r, v in zip(rounds, values) if v is not None]
                if valid_points:
                    valid_rounds, valid_values = zip(*valid_points)
                    plt.plot(valid_rounds, valid_values, marker='o', color=metric_color, 
                           linestyle='-', linewidth=2, label=metric_name.title())
            
            plt.xlabel("Ronda", fontsize=12, fontweight='bold')
            plt.ylabel("Valor", fontsize=12, fontweight='bold')
            plt.title("Evolução das Métricas de Classificação (Teste)", fontsize=14, fontweight='bold')
            
            min_val = min([min(history[m]) for m in available_metrics if m in history and len(history[m]) > 0])
            min_y = min(0.5, min_val * 0.95) if min_val < 0.9 else 0.9
            plt.ylim(min_y, 1.01)  # Ligeiramente acima de 1 para mostrar pontos no valor 1
            
            plt.legend(fontsize=12, framealpha=0.9)
            plt.xticks(rounds, fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plot_filename = "test_metrics_evolution.png"
            plot_path = base_reports / plot_filename
            plt.savefig(plot_path, dpi=120)
            plt.close()
            plot_files.append(plot_filename)
            logger.info(f"Generated plot: {plot_path}")
            
        # 5. GRÁFICO DE TEMPOS DE PROCESSAMENTO
        time_keys = ['fit_duration_seconds', 'evaluate_duration_seconds']
        explainability_keys = ['lime_duration_seconds', 'shap_duration_seconds']
        
        # 5.1 Gráfico de tempos regulares (fit, evaluate)
        if any(key in history for key in time_keys):
            plt.figure(figsize=(10, 6))
            
            available_metrics = [m for m in time_keys if m in history and len(history[m]) > 0]
            
            max_len = max([len(history[m]) for m in available_metrics]) if available_metrics else 0
            rounds = list(range(1, max_len + 1))
            
            for metric in available_metrics:
                # Extrair o nome da métrica sem o sufixo '_duration_seconds'
                metric_name = metric.replace('_duration_seconds', '')
                metric_color = colors.get(metric_name, '#333333')
                
                values = history[metric]
                if len(values) < max_len:
                    values = values + [None] * (max_len - len(values))
                
                valid_points = [(r, v) for r, v in zip(rounds, values) if v is not None]
                if valid_points:
                    valid_rounds, valid_values = zip(*valid_points)
                    plt.plot(valid_rounds, valid_values, marker='o', color=metric_color, 
                           linestyle='-', linewidth=2, label=metric_name.title())
            
            plt.xlabel("Ronda", fontsize=12, fontweight='bold')
            plt.ylabel("Tempo (segundos)", fontsize=12, fontweight='bold')
            plt.title("Tempos de Processamento por Ronda", fontsize=14, fontweight='bold')
            plt.legend(fontsize=12, framealpha=0.9)
            plt.xticks(rounds, fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plot_filename = "processing_times_evolution.png"
            plot_path = base_reports / plot_filename
            plt.savefig(plot_path, dpi=120)
            plt.close()
            plot_files.append(plot_filename)
            logger.info(f"Generated plot: {plot_path}")
        
        # 5.2 Gráfico de tempos de explicabilidade (lime, shap - barras)
        if any(key in history for key in explainability_keys) and any(history.get(key, [0])[-1] > 0 for key in explainability_keys):
            plt.figure(figsize=(10, 6))
            
            available_metrics = [m for m in explainability_keys if m in history and any(v > 0 for v in history[m])]
            
            # Para explicabilidade, geralmente só temos valores na última ronda
            # Vamos extrair os valores não-zero de cada métrica
            data = {}
            for metric in available_metrics:
                metric_name = metric.replace('_duration_seconds', '')
                non_zero_values = [v for v in history[metric] if v > 0]
                if non_zero_values:
                    data[metric_name] = non_zero_values[-1]  # Pegar o último valor não-zero
            
            # Criar gráfico de barras
            if data:
                names = list(data.keys())
                values = list(data.values())
                
                bars = plt.bar(names, values, color=[colors.get(name, '#333333') for name in names])
                
                # Adicionar valores acima das barras
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
                
                plt.xlabel("Método de Explicabilidade", fontsize=12, fontweight='bold')
                plt.ylabel("Tempo (segundos)", fontsize=12, fontweight='bold')
                plt.title("Tempos de Processamento das Explicabilidades", fontsize=14, fontweight='bold')
                plt.xticks(fontsize=10)
                plt.yticks(fontsize=10)
                plt.grid(True, axis='y', linestyle='--', alpha=0.7)
                
                plt.tight_layout()
                plot_filename = "explainability_times.png"
                plot_path = base_reports / plot_filename
                plt.savefig(plot_path, dpi=120)
                plt.close()
                # Não adicionamos explainability_times.png à lista de plot_files
                # para evitar que apareça na seção de evolução de métricas
                logger.info(f"Generated plot: {plot_path}")
                
        # 6. GRÁFICO DE COMPARAÇÃO VALIDAÇÃO vs TESTE (ACC)
        if all(key in history for key in ['val_accuracy', 'test_accuracy']):
            plt.figure(figsize=(10, 6))
            
            # Encontrar o número máximo de rounds para cada métrica
            val_acc_len = len(history['val_accuracy'])
            test_acc_len = len(history['test_accuracy'])
            
            # Criar eixos x para cada série
            val_rounds = list(range(1, val_acc_len + 1))
            test_rounds = list(range(1, test_acc_len + 1))
            
            plt.plot(val_rounds, history['val_accuracy'], marker='o', color=colors['val'], 
                   linestyle='-', linewidth=2, label='Validação')
            plt.plot(test_rounds, history['test_accuracy'], marker='^', color=colors['test'], 
                   linestyle='-', linewidth=2, label='Teste')
            
            plt.xlabel("Ronda", fontsize=12, fontweight='bold')
            plt.ylabel("Acurácia", fontsize=12, fontweight='bold')
            plt.title("Comparação de Acurácia: Validação vs Teste", fontsize=14, fontweight='bold')
            
            # Definir intervalo do eixo y para mostrar diferenças sutis
            # Ou fazer zoom em torno dos valores reais se forem todos muito altos
            plt.ylim(0.9, 1.01)  # Assumindo que as acurácias são altas
            
            plt.legend(fontsize=12, framealpha=0.9)
            plt.xticks(range(1, max(val_acc_len, test_acc_len) + 1), fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plot_filename = "accuracy_comparison.png"
            plot_path = base_reports / plot_filename
            plt.savefig(plot_path, dpi=120)
            plt.close()
            plot_files.append(plot_filename)
            logger.info(f"Generated plot: {plot_path}")
    
    except Exception as e:
        logger.error(f"Error generating enhanced plots: {e}", exc_info=True)
        
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
            /* Estilos para o novo layout flexível dos gráficos de evolução */
            .evolution-plots {{ 
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                justify-content: center;
            }}
            .plot-item {{ 
                flex: 1 1 400px;
                max-width: 100%;
                background-color: #fff;
                padding: 15px;
                border-radius: 4px;
                box-shadow: 0 1px 2px rgba(0,0,0,0.05);
                border: 1px solid #eee;
                text-align: center;
                margin-bottom: 20px;
            }}
            .plot-item img {{ 
                max-width: 100%;
                height: auto;
                margin: 10px auto;
                display: block;
            }}
            .plot-item h3 {{ 
                margin-top: 0;
                border-bottom: 1px solid #eee;
                padding-bottom: 5px;
                margin-bottom: 10px;
                color: #2c3e50;
            }}
            /* Estilos para a tabela de plots - mantido para compatibilidade */
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
        # Agrupar as métricas por tipo
        training_metrics = ["train_loss"]
        validation_metrics = ["val_loss", "val_rmse", "val_accuracy", "val_precision", "val_recall", "val_f1"]
        test_metrics = ["test_loss", "test_rmse", "test_accuracy", "test_precision", "test_recall", "test_f1"]
        time_metrics = ["fit_duration_seconds", "evaluate_duration_seconds", "lime_duration_seconds", "shap_duration_seconds"]
        
        # 1. Primeiro, tabela de métricas de treino e validação
        html_content += '<h3>Tabela 1: Métricas de Treino e Validação</h3>'
        html_content += '<div class="table-container">\n<table class="metrics-table">\n<thead>\n<tr><th>Ronda</th>'
        
        for key in training_metrics + validation_metrics:
            if key in history:
                html_content += f"<th>{key.replace('_', ' ').title()}</th>"
        html_content += "</tr>\n</thead>\n<tbody>\n"

        for r in range(num_rounds_completed):
            html_content += f"<tr><td>{r + 1}</td>"
            for key in training_metrics + validation_metrics:
                if key in history:
                    metric_list = history.get(key, [])
                    value = metric_list[r] if r < len(metric_list) else None
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        formatted_value = "N/A"
                    elif isinstance(value, (int, float)):
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    html_content += f"<td>{formatted_value}</td>"
            html_content += "</tr>\n"
        html_content += "</tbody>\n</table>\n</div>"
        
        # 2. Segunda tabela: métricas de teste
        html_content += '<h3>Tabela 2: Métricas de Teste</h3>'
        html_content += '<div class="table-container">\n<table class="metrics-table">\n<thead>\n<tr><th>Ronda</th>'
        
        for key in test_metrics:
            if key in history:
                html_content += f"<th>{key.replace('_', ' ').title()}</th>"
        html_content += "</tr>\n</thead>\n<tbody>\n"

        for r in range(num_rounds_completed):
            html_content += f"<tr><td>{r + 1}</td>"
            for key in test_metrics:
                if key in history:
                    metric_list = history.get(key, [])
                    value = metric_list[r] if r < len(metric_list) else None
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        formatted_value = "N/A"
                    elif isinstance(value, (int, float)):
                        formatted_value = f"{value:.4f}"
                    else:
                        formatted_value = str(value)
                    html_content += f"<td>{formatted_value}</td>"
            html_content += "</tr>\n"
        html_content += "</tbody>\n</table>\n</div>"
        
        # 3. Terceira tabela: métricas de tempo
        html_content += '<h3>Tabela 3: Tempos de Processamento (segundos)</h3>'
        html_content += '<div class="table-container">\n<table class="metrics-table">\n<thead>\n<tr><th>Ronda</th>'
        
        for key in time_metrics:
            if key in history:
                html_content += f"<th>{key.replace('_duration_seconds', '').replace('_', ' ').title()}</th>"
        html_content += "</tr>\n</thead>\n<tbody>\n"

        for r in range(num_rounds_completed):
            html_content += f"<tr><td>{r + 1}</td>"
            for key in time_metrics:
                if key in history:
                    metric_list = history.get(key, [])
                    value = metric_list[r] if r < len(metric_list) else None
                    if value is None or (isinstance(value, float) and np.isnan(value)):
                        formatted_value = "N/A"
                    elif isinstance(value, (int, float)):
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
        html_content += '<div class="evolution-plots">\n'
        for plot_file in plot_files:
            plot_title = plot_file.replace('_evolution.png','').replace('.png','').replace('_', ' ').title()
            html_content += f'<div class="plot-item">\n'
            html_content += f'<h3>{plot_title}</h3>\n'
            html_content += f'<img src="{plot_file}" alt="{plot_title}" />\n'
            html_content += '</div>\n'
        html_content += '</div>\n'
    else:
         html_content += "<p><em>Nenhum gráfico de evolução foi gerado ou encontrado.</em></p>\n"
    html_content += "</div>" # Fim da section dos plots


    # Secção de Explicabilidade (mantém-se, usa os ficheiros finais gerados antes)
    html_content += f"""
        <div class="section">
            <h2>Explicabilidade (Resultados Finais da Última Ronda)</h2>
            
            <!-- Adicionar gráfico de tempos de explicabilidade aqui -->
            {f'<h3>Tempos de Explicabilidade</h3><img src="explainability_times.png" alt="Tempos de Explicabilidade" style="max-width: 600px;"/>' if 'explainability_times.png' in os.listdir(base_reports) else ''}
            
            <!-- Layout de duas colunas para LIME e SHAP -->
            <div style="display: flex; flex-wrap: wrap; gap: 20px; justify-content: space-between; margin-top: 20px;">
                <!-- Coluna 1: LIME -->
                <div style="flex: 1; min-width: 300px;">
                    <h3>LIME</h3>
                    
                    <!-- Informações sobre a instância analisada (novo) -->
                    <div style="background-color: #f8f9fa; padding: 10px; border-left: 3px solid #1abc9c; margin-bottom: 15px;">
                        <h4>Instância Analisada</h4>
                        {lime_instance_html}
                    </div>
                    
                    {f'<img src="lime_final.png" alt="LIME final" style="width: 100%; max-width: 600px;"/>' if 'lime_final.png' in os.listdir(base_reports) else '<p><em>Explicação LIME não disponível ou não gerada.</em></p>'}
                    <p><a href="lime_explanation.txt" target="_blank">Ver LIME (texto)</a></p>
                    <p><a href="lime_explanation_formatado.txt" target="_blank">Ver LIME (formatado)</a></p>
                </div>
                
                <!-- Coluna 2: SHAP -->
                <div style="flex: 1; min-width: 300px;">
                    <h3>SHAP</h3>
                    {f'<img src="shap_final.png" alt="SHAP final" style="width: 100%; max-width: 600px;"/>' if 'shap_final.png' in os.listdir(base_reports) else '<p><em>Explicação SHAP não disponível ou não gerada.</em></p>'}
                    <p><a href="shap_values.npy" download>Download SHAP Values (.npy)</a></p>
                </div>
            </div>
            
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
                {f'<li><a href="lime_explanation_formatado.txt" target="_blank">Explicação LIME (formatado)</a></li>' if 'lime_explanation_formatado.txt' in os.listdir(base_reports) else ''}
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

def generate_explainability(explainer, X_train, base_reports, sample_idx=None):
    """
    Gera explicabilidade LIME e SHAP para o modelo.
    
    Args:
        explainer: Instância de ModelExplainer
        X_train: Dados de treino ou caminho para um arquivo .npy contendo os dados
        base_reports: Diretório base para salvar as explicações
        sample_idx: Índice da amostra para explicar com LIME. Se None, seleciona aleatoriamente.
        
    Returns:
        tuple: Tupla com durações (lime_duration, shap_duration)
    """
    lime_duration = 0
    shap_duration = 0
    
    try:
        # Verificar se X_train é um caminho para arquivo e carregar os dados se necessário
        if isinstance(X_train, (str, Path)):
            logger.info(f"Carregando X_train do arquivo: {X_train}")
            X_train_path = Path(X_train)
            if X_train_path.exists():
                X_train = np.load(X_train_path)
                logger.info(f"Dados X_train carregados com sucesso: {X_train.shape}")
            else:
                logger.error(f"Arquivo X_train não encontrado em {X_train_path}")
                return lime_duration, shap_duration
        
        # Gerar LIME
        lime_start = time.time()
        try:
            # Escolher instância para explicar
            if sample_idx is None:
                # Selecionar instância aleatória
                import random
                random_seed = 42 + int(time.time() * 1000) % 10000
                random.seed(random_seed)
                sample_idx = random.randint(0, len(X_train) - 1)
                logger.info(f'Selecionada instância aleatória com índice {sample_idx}')
            else:
                logger.info(f'Usando instância com índice {sample_idx}')
            
            # Obter a instância para explicar
            instance = X_train[sample_idx].reshape(1, -1)[0]
            
            # Obter previsão do modelo para esta instância
            instance_tensor = torch.tensor(instance, dtype=torch.float32).to(explainer.device)
            with torch.no_grad():
                prediction = explainer.model(instance_tensor.unsqueeze(0)).item()
            
            # Tentar carregar o dataset original para obter valores não normalizados
            try:
                # Caminho para o dataset original (ajuste conforme necessário)
                original_dataset_path = '/app/DatasetIOT/transformed_dataset_imeisv_8642840401612300.csv'
                if os.path.exists(original_dataset_path):
                    logger.info(f"Carregando valores originais de {original_dataset_path}")
                    import pandas as pd
                    df_original = pd.read_csv(original_dataset_path)
                    
                    # Extrair features temporais do timestamp
                    if '_time' in df_original.columns:
                        logger.info(f"Extraindo features temporais de _time para valores originais")
                        # Converter para datetime
                        try:
                            df_original['_time'] = pd.to_datetime(df_original['_time'])
                        except:
                            try:
                                logger.info("Standard datetime parsing failed, trying with ISO8601 format")
                                df_original['_time'] = pd.to_datetime(df_original['_time'], format='ISO8601')
                            except Exception as e:
                                logger.error(f"Failed to parse _time column as datetime: {e}")
                        
                        # Extrair features temporais
                        if pd.api.types.is_datetime64_any_dtype(df_original['_time']):
                            df_original['hour'] = df_original['_time'].dt.hour
                            df_original['dayofweek'] = df_original['_time'].dt.dayofweek
                            df_original['day'] = df_original['_time'].dt.day
                            df_original['month'] = df_original['_time'].dt.month
                            df_original['is_weekend'] = (df_original['dayofweek'] >= 5).astype(int)
                            logger.info("Features temporais extraídas com sucesso do dataset original")
                    
                    # Se tivermos o índice original, usamos para buscar os valores originais
                    original_instance = df_original.iloc[sample_idx]
                    
                    # Obter valor target original se disponível
                    original_target = None
                    original_target_class = None
                    prediction_correct = None
                    
                    if 'attack' in df_original.columns:
                        original_target = int(original_instance['attack'])
                        original_target_class = "Ataque" if original_target == 1 else "Normal"
                        # Determinar se a previsão está correta
                        prediction_class_numeric = 1 if prediction >= 0.5 else 0
                        prediction_correct = (prediction_class_numeric == original_target)
                        logger.info(f"Valor target original: {original_target} ({original_target_class}), Previsão correta: {prediction_correct}")
                    
                    # Remover colunas que não são features
                    cols_to_drop = []
                    for col in ['indice', 'imeisv', 'index', 'attack', '_time']:
                        if col in original_instance.index:
                            cols_to_drop.append(col)
                    
                    original_features = original_instance.drop(cols_to_drop) if cols_to_drop else original_instance
                    
                    # Extrair informações sobre a instância
                    instance_info = {
                        "index": sample_idx,
                        "original_index": sample_idx,  # Adicionando para compatibilidade com o relatório HTML
                        "prediction": round(prediction, 5),
                        "predicted_class": "Ataque" if prediction >= 0.5 else "Normal",
                        "prediction_confidence": round(max(prediction, 1-prediction) * 100, 2),
                        "feature_count": len(instance),
                        # Adicionar informações sobre o target original
                        "original_target": original_target,
                        "original_target_class": original_target_class,
                        "prediction_correct": prediction_correct,
                        "features": {}
                    }
                    
                    # Adicionar features originais ao dicionário
                    for i, name in enumerate(explainer.feature_names):
                        if name in ['hour', 'dayofweek', 'day', 'month', 'is_weekend'] and name in df_original.columns:
                            # Para features temporais, usar os valores extraídos (valores inteiros - não normalizados)
                            instance_info["features"][name] = int(original_instance[name])
                        elif name in original_features:
                            # Usar valor original do dataset para outras features
                            instance_info["features"][name] = float(original_features[name])
                        else:
                            # Para features não encontradas no dataset original, usar o valor normalizado
                            instance_info["features"][name] = float(instance[i])
                    
                    logger.info(f"Valores originais carregados com sucesso para {len(instance_info['features'])} features")
                else:
                    # Se não encontrar o dataset original, usar os valores normalizados
                    logger.warning(f"Dataset original não encontrado em {original_dataset_path}, usando valores normalizados")
                    instance_info = {
                        "index": sample_idx,
                        "original_index": sample_idx,  # Adicionando para compatibilidade com o relatório HTML
                        "prediction": round(prediction, 5),
                        "predicted_class": "Ataque" if prediction >= 0.5 else "Normal",
                        "prediction_confidence": round(max(prediction, 1-prediction) * 100, 2),
                        "feature_count": len(instance),
                        # Adicionar informações sobre o target original
                        "original_target": None,
                        "original_target_class": None,
                        "prediction_correct": None,
                        "features": {
                            name: round(float(value), 5)
                            for name, value in zip(explainer.feature_names, instance)
                        }
                    }
            except Exception as e:
                logger.error(f"Erro ao carregar valores originais: {e}")
                # Fallback para os valores normalizados em caso de erro
                instance_info = {
                    "index": sample_idx,
                    "original_index": sample_idx,  # Adicionando para compatibilidade com o relatório HTML
                    "prediction": round(prediction, 5),
                    "predicted_class": "Ataque" if prediction >= 0.5 else "Normal",
                    "prediction_confidence": round(max(prediction, 1-prediction) * 100, 2),
                    "feature_count": len(instance),
                    # Adicionar informações sobre o target original
                    "original_target": None,
                    "original_target_class": None,
                    "prediction_correct": None,
                    "features": {
                        name: round(float(value), 5)
                        for name, value in zip(explainer.feature_names, instance)
                    }
                }
            
            # Salvar informações da instância como JSON
            instance_info_path = base_reports / "lime_instance_info.json"
            with open(instance_info_path, "w") as f:
                json.dump(instance_info, f, indent=4, default=str)
            
            # Executar LIME
            lime_exp = explainer.explain_lime(X_train, instance)
            
            # Salvar explicação LIME como texto
            with open(base_reports / 'lime_explanation.txt', 'w', encoding='utf-8') as f:
                f.write(str(lime_exp.as_list()))
                
            # Criar uma versão formatada do arquivo de explicação LIME
            try:
                with open(base_reports / 'lime_explanation.txt', 'r', encoding='utf-8') as f_in:
                    lime_text = f_in.read()
                
                # Formatação mais legível
                formatted_explanation = ""
                formatted_explanation += "# Explicação LIME\n\n"
                
                # Informações da instância
                formatted_explanation += f"## Amostra {sample_idx}\n\n"
                formatted_explanation += "### Previsão do Modelo\n"
                formatted_explanation += f"- **Classe Prevista:** {'ATAQUE' if prediction >= 0.5 else 'NORMAL'}\n"
                formatted_explanation += f"- **Probabilidade:** {prediction:.4f}\n\n"
                
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
                    formatted_explanation += f"### {category}\n\n"
                    
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
                    
                    formatted_explanation += header + "\n"
                    formatted_explanation += separator + "\n"
                    
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
            lime_png_path = base_reports / 'lime_final.png'
            
            # Não gerar o arquivo HTML indesejado
            # lime_exp.save_to_file(str(base_reports / 'lime_final.html'))
            
            # Usar nossa visualização personalizada
            explainer.create_custom_lime_visualization(lime_exp, lime_png_path)
            
            logger.info(f'LIME explanation generated and saved to {lime_png_path}')
        except Exception as e:
            logger.error(f'Error generating LIME explanation: {e}', exc_info=True)
        finally:
            lime_duration = time.time() - lime_start
        
        # Gerar SHAP
        shap_start = time.time()
        try:
            # Executar SHAP
            shap_values = explainer.explain_shap(X_train)
            
            # Salvar valores SHAP como arquivo .npy
            shap_values_path = base_reports / 'shap_values.npy'
            np.save(shap_values_path, shap_values)
            
            # Gerar visualização SHAP
            shap_png_path = base_reports / 'shap_final.png'
            
            try:
                # Verificar se o método personalizado existe
                if hasattr(explainer, 'create_custom_shap_visualization'):
                    explainer.create_custom_shap_visualization(shap_values, shap_png_path)
                else:
                    # Criar visualização SHAP manualmente se o método não existir
                    plt.figure(figsize=(12, 8))
                    
                    # Calcular importância média das features
                    if isinstance(shap_values, list):
                        feature_importance = np.abs(shap_values[1]).mean(0)
                    else:
                        feature_importance = np.abs(shap_values).mean(0)
                    
                    # Verificar dimensionalidade
                    if feature_importance.ndim > 1:
                        feature_importance = feature_importance.mean(axis=1)
                    
                    # Ordenar features por importância
                    indices = np.argsort(feature_importance)
                    n_features = min(20, len(feature_importance))  # Limitar a 20 features
                    plt.barh(range(n_features), feature_importance[indices[-n_features:]], color='#1E88E5')
                    plt.yticks(range(n_features), [explainer.feature_names[i] for i in indices[-n_features:]])
                    plt.xlabel('Importância Média (|SHAP|)')
                    plt.title('Importância das Features (SHAP)')
                    plt.tight_layout()
                    plt.savefig(shap_png_path, dpi=150, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                logger.error(f"Erro ao criar visualização SHAP: {e}")
                # Criar um gráfico simples como fallback
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, "Não foi possível gerar visualização SHAP", 
                        horizontalalignment='center', verticalalignment='center', fontsize=14)
                plt.savefig(shap_png_path)
                plt.close()
            
            logger.info(f'SHAP explanation generated and saved to {shap_png_path}')
        except Exception as e:
            logger.error(f'Error generating SHAP explanation: {e}', exc_info=True)
        finally:
            shap_duration = time.time() - shap_start
        
        return lime_duration, shap_duration
    except Exception as e:
        logger.error(f'Error in generate_explainability: {e}', exc_info=True)
        return 0, 0

def generate_explainability_formatado(explainer, X_train, base_reports, sample_idx=None):
    """
    Gera explicabilidade LIME formatada para o modelo.
    
    Args:
        explainer: Instância de ModelExplainer
        X_train: Dados de treino ou caminho para um arquivo .npy contendo os dados
        base_reports: Diretório base para salvar as explicações
        sample_idx: Índice da amostra para explicar com LIME. Se None, seleciona aleatoriamente.
        
    Returns:
        tuple: Tupla com durações (lime_duration, shap_duration)
    """
    lime_duration = 0
    shap_duration = 0
    
    try:
        # Verificar se X_train é um caminho para arquivo e carregar os dados se necessário
        if isinstance(X_train, (str, Path)):
            logger.info(f"Carregando X_train do arquivo: {X_train}")
            X_train_path = Path(X_train)
            if X_train_path.exists():
                X_train = np.load(X_train_path)
                logger.info(f"Dados X_train carregados com sucesso: {X_train.shape}")
            else:
                logger.error(f"Arquivo X_train não encontrado em {X_train_path}")
                return lime_duration, shap_duration
        
        # Gerar LIME
        lime_start = time.time()
        try:
            # Escolher instância para explicar
            if sample_idx is None:
                # Selecionar instância aleatória
                import random
                random_seed = 42 + int(time.time() * 1000) % 10000
                random.seed(random_seed)
                sample_idx = random.randint(0, len(X_train) - 1)
                logger.info(f'Selecionada instância aleatória com índice {sample_idx}')
            else:
                logger.info(f'Usando instância com índice {sample_idx}')
            
            # Obter a instância para explicar
            instance = X_train[sample_idx].reshape(1, -1)[0]
            
            # Obter previsão do modelo para esta instância
            instance_tensor = torch.tensor(instance, dtype=torch.float32).to(explainer.device)
            with torch.no_grad():
                prediction = explainer.model(instance_tensor.unsqueeze(0)).item()
            
            # Tentar carregar o dataset original para obter valores não normalizados
            try:
                # Caminho para o dataset original (ajuste conforme necessário)
                original_dataset_path = '/app/DatasetIOT/transformed_dataset_imeisv_8642840401612300.csv'
                if os.path.exists(original_dataset_path):
                    logger.info(f"Carregando valores originais de {original_dataset_path}")
                    import pandas as pd
                    df_original = pd.read_csv(original_dataset_path)
                    
                    # Extrair features temporais do timestamp
                    if '_time' in df_original.columns:
                        logger.info(f"Extraindo features temporais de _time para valores originais")
                        # Converter para datetime
                        try:
                            df_original['_time'] = pd.to_datetime(df_original['_time'])
                        except:
                            try:
                                logger.info("Standard datetime parsing failed, trying with ISO8601 format")
                                df_original['_time'] = pd.to_datetime(df_original['_time'], format='ISO8601')
                            except Exception as e:
                                logger.error(f"Failed to parse _time column as datetime: {e}")
                        
                        # Extrair features temporais
                        if pd.api.types.is_datetime64_any_dtype(df_original['_time']):
                            df_original['hour'] = df_original['_time'].dt.hour
                            df_original['dayofweek'] = df_original['_time'].dt.dayofweek
                            df_original['day'] = df_original['_time'].dt.day
                            df_original['month'] = df_original['_time'].dt.month
                            df_original['is_weekend'] = (df_original['dayofweek'] >= 5).astype(int)
                            logger.info("Features temporais extraídas com sucesso do dataset original")
                    
                    # Se tivermos o índice original, usamos para buscar os valores originais
                    original_instance = df_original.iloc[sample_idx]
                    
                    # Obter valor target original se disponível
                    original_target = None
                    original_target_class = None
                    prediction_correct = None
                    
                    if 'attack' in df_original.columns:
                        original_target = int(original_instance['attack'])
                        original_target_class = "Ataque" if original_target == 1 else "Normal"
                        # Determinar se a previsão está correta
                        prediction_class_numeric = 1 if prediction >= 0.5 else 0
                        prediction_correct = (prediction_class_numeric == original_target)
                        logger.info(f"Valor target original: {original_target} ({original_target_class}), Previsão correta: {prediction_correct}")
                    
                    # Remover colunas que não são features
                    cols_to_drop = []
                    for col in ['indice', 'imeisv', 'index', 'attack', '_time']:
                        if col in original_instance.index:
                            cols_to_drop.append(col)
                    
                    original_features = original_instance.drop(cols_to_drop) if cols_to_drop else original_instance
                    
                    # Extrair informações sobre a instância
                    instance_info = {
                        "index": sample_idx,
                        "original_index": sample_idx,  # Adicionando para compatibilidade com o relatório HTML
                        "prediction": round(prediction, 5),
                        "predicted_class": "Ataque" if prediction >= 0.5 else "Normal",
                        "prediction_confidence": round(max(prediction, 1-prediction) * 100, 2),
                        "feature_count": len(instance),
                        # Adicionar informações sobre o target original
                        "original_target": original_target,
                        "original_target_class": original_target_class,
                        "prediction_correct": prediction_correct,
                        "features": {}
                    }
                    
                    # Adicionar features originais ao dicionário
                    for i, name in enumerate(explainer.feature_names):
                        if name in ['hour', 'dayofweek', 'day', 'month', 'is_weekend'] and name in df_original.columns:
                            # Para features temporais, usar os valores extraídos (valores inteiros - não normalizados)
                            instance_info["features"][name] = int(original_instance[name])
                        elif name in original_features:
                            # Usar valor original do dataset para outras features
                            instance_info["features"][name] = float(original_features[name])
                        else:
                            # Para features não encontradas no dataset original, usar o valor normalizado
                            instance_info["features"][name] = float(instance[i])
                    
                    logger.info(f"Valores originais carregados com sucesso para {len(instance_info['features'])} features")
                else:
                    # Se não encontrar o dataset original, usar os valores normalizados
                    logger.warning(f"Dataset original não encontrado em {original_dataset_path}, usando valores normalizados")
                    instance_info = {
                        "index": sample_idx,
                        "original_index": sample_idx,  # Adicionando para compatibilidade com o relatório HTML
                        "prediction": round(prediction, 5),
                        "predicted_class": "Ataque" if prediction >= 0.5 else "Normal",
                        "prediction_confidence": round(max(prediction, 1-prediction) * 100, 2),
                        "feature_count": len(instance),
                        # Adicionar informações sobre o target original
                        "original_target": None,
                        "original_target_class": None,
                        "prediction_correct": None,
                        "features": {
                            name: round(float(value), 5)
                            for name, value in zip(explainer.feature_names, instance)
                        }
                    }
            except Exception as e:
                logger.error(f"Erro ao carregar valores originais: {e}")
                # Fallback para os valores normalizados em caso de erro
                instance_info = {
                    "index": sample_idx,
                    "original_index": sample_idx,  # Adicionando para compatibilidade com o relatório HTML
                    "prediction": round(prediction, 5),
                    "predicted_class": "Ataque" if prediction >= 0.5 else "Normal",
                    "prediction_confidence": round(max(prediction, 1-prediction) * 100, 2),
                    "feature_count": len(instance),
                    # Adicionar informações sobre o target original
                    "original_target": None,
                    "original_target_class": None,
                    "prediction_correct": None,
                    "features": {
                        name: round(float(value), 5)
                        for name, value in zip(explainer.feature_names, instance)
                    }
                }
            
            # Salvar informações da instância como JSON
            instance_info_path = base_reports / "lime_instance_info.json"
            with open(instance_info_path, "w") as f:
                json.dump(instance_info, f, indent=4, default=str)
            
            # Executar LIME
            lime_exp = explainer.explain_lime(X_train, instance)
            
            # Salvar explicação LIME como texto
            with open(base_reports / 'lime_explanation.txt', 'w', encoding='utf-8') as f:
                f.write(str(lime_exp.as_list()))
                
            # Criar uma versão formatada do arquivo de explicação LIME
            try:
                with open(base_reports / 'lime_explanation.txt', 'r', encoding='utf-8') as f_in:
                    lime_text = f_in.read()
                
                # Formatação mais legível
                formatted_explanation = ""
                formatted_explanation += "# Explicação LIME\n\n"
                
                # Informações da instância
                formatted_explanation += f"## Amostra {sample_idx}\n\n"
                formatted_explanation += "### Previsão do Modelo\n"
                formatted_explanation += f"- **Classe Prevista:** {'ATAQUE' if prediction >= 0.5 else 'NORMAL'}\n"
                formatted_explanation += f"- **Probabilidade:** {prediction:.4f}\n\n"
                
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
                    "Retransmissões DL (cell_x_dl_retx)": [],
                    "Transmissões DL (cell_x_dl_tx)": [],
                    "Retransmissões UL (cell_x_ul_retx)": [],
                    "Transmissões UL (cell_x_ul_tx)": [],
                    "Bytes de Upload (não incrementais)": [],
                    "Bytes de Download (não incrementais)": [],
                    "Bytes de Upload (incrementais)": [],
                    "Bytes de Download (incrementais)": [],
                    "Pacotes de Upload": [],
                    "Pacotes de Download": [],
                    "Features Temporais": [],
                    "Outras Features": []
                }
                
                # Preencher features por categoria (todas as features, não apenas as explicadas)
                for feature_name, value in all_features.items():
                    # Obter contribuição (se existir)
                    contribution = explained_features.get(feature_name, 0)
                    
                    if feature_name.startswith("cell_x_dl_retx"):
                        categories["Retransmissões DL (cell_x_dl_retx)"].append((feature_name, value, contribution))
                    elif feature_name.startswith("cell_x_dl_tx"):
                        categories["Transmissões DL (cell_x_dl_tx)"].append((feature_name, value, contribution))
                    elif feature_name.startswith("cell_x_ul_retx"):
                        categories["Retransmissões UL (cell_x_ul_retx)"].append((feature_name, value, contribution))
                    elif feature_name.startswith("cell_x_ul_tx"):
                        categories["Transmissões UL (cell_x_ul_tx)"].append((feature_name, value, contribution))
                    elif "dl_bitrate" in feature_name:
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
                    elif feature_name in ["hour", "dayofweek", "day", "month", "is_weekend"] or feature_name.endswith("_time"):
                        categories["Features Temporais"].append((feature_name, value, contribution))
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
                    formatted_explanation += f"### {category}\n\n"
                    if not features:
                        formatted_explanation += "_Nenhuma feature nesta categoria._\n\n"
                        continue
                    
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
                    
                    formatted_explanation += header + "\n"
                    formatted_explanation += separator + "\n"
                    
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
            lime_png_path = base_reports / 'lime_final.png'
            
            # Não gerar o arquivo HTML indesejado
            # lime_exp.save_to_file(str(base_reports / 'lime_final.html'))
            
            # Usar nossa visualização personalizada
            explainer.create_custom_lime_visualization(lime_exp, lime_png_path)
            
            logger.info(f'LIME explanation generated and saved to {lime_png_path}')
        except Exception as e:
            logger.error(f'Error generating LIME explanation: {e}', exc_info=True)
        finally:
            lime_duration = time.time() - lime_start
        
        # Gerar SHAP
        shap_start = time.time()
        try:
            # Executar SHAP
            shap_values = explainer.explain_shap(X_train)
            
            # Salvar valores SHAP como arquivo .npy
            shap_values_path = base_reports / 'shap_values.npy'
            np.save(shap_values_path, shap_values)
            
            # Gerar visualização SHAP
            shap_png_path = base_reports / 'shap_final.png'
            
            try:
                # Verificar se o método personalizado existe
                if hasattr(explainer, 'create_custom_shap_visualization'):
                    explainer.create_custom_shap_visualization(shap_values, shap_png_path)
                else:
                    # Criar visualização SHAP manualmente se o método não existir
                    plt.figure(figsize=(12, 8))
                    
                    # Calcular importância média das features
                    if isinstance(shap_values, list):
                        feature_importance = np.abs(shap_values[1]).mean(0)
                    else:
                        feature_importance = np.abs(shap_values).mean(0)
                    
                    # Verificar dimensionalidade
                    if feature_importance.ndim > 1:
                        feature_importance = feature_importance.mean(axis=1)
                    
                    # Ordenar features por importância
                    indices = np.argsort(feature_importance)
                    n_features = min(20, len(feature_importance))  # Limitar a 20 features
                    plt.barh(range(n_features), feature_importance[indices[-n_features:]], color='#1E88E5')
                    plt.yticks(range(n_features), [explainer.feature_names[i] for i in indices[-n_features:]])
                    plt.xlabel('Importância Média (|SHAP|)')
                    plt.title('Importância das Features (SHAP)')
                    plt.tight_layout()
                    plt.savefig(shap_png_path, dpi=150, bbox_inches='tight')
                    plt.close()
            except Exception as e:
                logger.error(f"Erro ao criar visualização SHAP: {e}")
                # Criar um gráfico simples como fallback
                plt.figure(figsize=(10, 6))
                plt.text(0.5, 0.5, "Não foi possível gerar visualização SHAP", 
                        horizontalalignment='center', verticalalignment='center', fontsize=14)
                plt.savefig(shap_png_path)
                plt.close()
            
            logger.info(f'SHAP explanation generated and saved to {shap_png_path}')
        except Exception as e:
            logger.error(f'Error generating SHAP explanation: {e}', exc_info=True)
        finally:
            shap_duration = time.time() - shap_start
        
        return lime_duration, shap_duration
    except Exception as e:
        logger.error(f'Error in generate_explainability: {e}', exc_info=True)
        return 0, 0
