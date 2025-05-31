"""Helper functions for generating and saving plots for reports."""
import logging
import matplotlib.pyplot as plt
import numpy as np # Adicionado para np.arange e outras operações numpy
from pathlib import Path # Added for Path object type hint and usage
from typing import Dict, List, Any # Added for type hinting

logger = logging.getLogger(__name__) # Use module-specific logger

def generate_evolution_plots(history: Dict[str, List[Any]], base_reports: Path) -> List[str]:
    """
    Gera gráficos de evolução das métricas ao longo das rondas.
    
    Args:
        history: Dicionário com o histórico de métricas
        base_reports: Diretório base para salvar os gráficos (Path object)
        
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
            
            if 'test_loss' in history and history['test_loss']:
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
                (history.get('test_loss', []) if history.get('test_loss') else [])
            )
            if all_losses: # Check if all_losses is not empty
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
        if 'val_rmse' in history and history['val_rmse']:
            plt.figure(figsize=(10, 6))
            
            rounds = list(range(1, len(history['val_rmse']) + 1))
            
            plt.plot(rounds, history['val_rmse'], marker='s', color=colors['val'], 
                   linestyle='-', linewidth=2, label='Validação')
            
            if 'test_rmse' in history and history['test_rmse']:
                plt.plot(rounds, history['test_rmse'], marker='^', color=colors['test'], 
                       linestyle='-', linewidth=2, label='Teste')
            
            plt.xlabel("Ronda", fontsize=12, fontweight='bold')
            plt.ylabel("RMSE", fontsize=12, fontweight='bold')
            plt.title("Evolução do RMSE por Ronda", fontsize=14, fontweight='bold')
            plt.legend(fontsize=12, framealpha=0.9)
            plt.xticks(rounds, fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            all_rmse = (
                history['val_rmse'] +
                (history.get('test_rmse', []) if history.get('test_rmse') else [])
            )
            if all_rmse:
                min_rmse = min(all_rmse) * 0.9
                max_rmse = max(all_rmse) * 1.1
                plt.ylim(min_rmse, max_rmse)

            plt.tight_layout()
            plot_filename = "rmse_evolution.png"
            plot_path = base_reports / plot_filename
            plt.savefig(plot_path, dpi=120)
            plt.close()
            plot_files.append(plot_filename)
            logger.info(f"Generated plot: {plot_path}")
        
        # 3. GRÁFICO DE MÉTRICAS DE CLASSIFICAÇÃO (ACC, PREC, REC, F1) - Validação
        val_metric_keys = ['val_accuracy', 'val_precision', 'val_recall', 'val_f1']
        if any(key in history and history[key] for key in val_metric_keys):
            plt.figure(figsize=(10, 6))  # Tamanho similar ao original
            
            # Identificar métricas disponíveis
            available_metrics = [m for m in val_metric_keys if m in history and len(history[m]) > 0]
            
            # Encontrar o máximo de rondas
            max_len = max([len(history[m]) for m in available_metrics]) if available_metrics else 0
            rounds = list(range(1, max_len + 1))
            
            for metric in available_metrics:
                # Extrair a parte principal do nome da métrica (sem 'val_')
                metric_name = metric.replace('val_', '')
                metric_color = colors.get(metric_name, '#333333')
                
                # Obter valores e lidar com comprimentos diferentes
                values = history[metric]
                if len(values) < max_len:
                    values = values + [None] * (max_len - len(values))
                
                # Filtrar pontos válidos (não-None)
                valid_points = [(r, v) for r, v in zip(rounds, values) if v is not None]
                if valid_points:
                    valid_rounds, valid_values = zip(*valid_points)
                    plt.plot(valid_rounds, valid_values, marker='o', color=metric_color, 
                           linestyle='-', linewidth=2, label=f"{metric_name.title()} (Validação)")
            
            plt.xlabel("Ronda", fontsize=12, fontweight='bold')
            plt.ylabel("Valor", fontsize=12, fontweight='bold')
            plt.title("Evolução das Métricas de Classificação (Validação)", fontsize=14, fontweight='bold')
            
            # Ajustar limites de Y como no original
            min_val = min([min(history[m]) for m in available_metrics if m in history and len(history[m]) > 0])
            min_y = min(0.5, min_val * 0.95) if min_val < 0.9 else 0.9
            plt.ylim(min_y, 1.01)  # Ligeiramente acima de 1 para mostrar pontos no valor 1
            
            plt.legend(fontsize=12, framealpha=0.9)  # Simples, sem bbox_to_anchor
            plt.xticks(rounds, fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()  # Sem ajuste rect
            plot_filename = "validation_metrics_evolution.png"
            plot_path = base_reports / plot_filename
            plt.savefig(plot_path, dpi=120)
            plt.close()
            plot_files.append(plot_filename)
            logger.info(f"Generated plot: {plot_path}")

        # 4. GRÁFICO DE MÉTRICAS DE TESTE - accuracy, precision, recall, f1
        test_metric_keys = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        if any(key in history and history[key] for key in test_metric_keys):
            plt.figure(figsize=(10, 6))  # Tamanho similar ao original
            
            # Identificar métricas disponíveis
            available_metrics = [m for m in test_metric_keys if m in history and len(history[m]) > 0]
            
            # Encontrar o máximo de rondas
            max_len = max([len(history[m]) for m in available_metrics]) if available_metrics else 0
            rounds = list(range(1, max_len + 1))
            
            for metric in available_metrics:
                # Extrair a parte principal do nome da métrica (sem 'test_')
                metric_name = metric.replace('test_', '')
                metric_color = colors.get(metric_name, '#333333')
                
                # Obter valores e lidar com comprimentos diferentes
                values = history[metric]
                if len(values) < max_len:
                    values = values + [None] * (max_len - len(values))
                
                # Filtrar pontos válidos (não-None)
                valid_points = [(r, v) for r, v in zip(rounds, values) if v is not None]
                if valid_points:
                    valid_rounds, valid_values = zip(*valid_points)
                    plt.plot(valid_rounds, valid_values, marker='o', color=metric_color, 
                           linestyle='-', linewidth=2, label=f"{metric_name.title()} (Teste)")
            
            plt.xlabel("Ronda", fontsize=12, fontweight='bold')
            plt.ylabel("Valor", fontsize=12, fontweight='bold')
            plt.title("Evolução das Métricas de Classificação (Teste)", fontsize=14, fontweight='bold')
            
            # Ajustar limites de Y como no original
            min_val = min([min(history[m]) for m in available_metrics if m in history and len(history[m]) > 0])
            min_y = min(0.5, min_val * 0.95) if min_val < 0.9 else 0.9
            plt.ylim(min_y, 1.01)  # Ligeiramente acima de 1 para mostrar pontos no valor 1
            
            plt.legend(fontsize=12, framealpha=0.9)  # Simples, sem bbox_to_anchor
            plt.xticks(rounds, fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            
            plt.tight_layout()  # Sem ajuste rect
            plot_filename = "test_metrics_evolution.png"
            plot_path = base_reports / plot_filename
            plt.savefig(plot_path, dpi=120)
            plt.close()
            plot_files.append(plot_filename)
            logger.info(f"Generated plot: {plot_path}")

        # 5. GRÁFICO DE TEMPO POR RONDA - Apenas Fit e Evaluate (sem LIME e SHAP)
        # Mapeamento de nomes de chaves para compatibilidade com versões antigas
        time_key_mapping = {
            'fit_duration_seconds': 'fit_time',
            'evaluate_duration_seconds': 'evaluate_time' 
            # Removidos lime_duration_seconds e shap_duration_seconds pois não queremos no gráfico
        }
        
        # Cria dicionário temporário com as duas versões das chaves
        combined_history = history.copy()
        for old_key, new_key in time_key_mapping.items():
            if old_key in history:
                combined_history[new_key] = history[old_key]
        
        # Usar nova lista de chaves apenas com fit e evaluate
        time_keys = ['fit_time', 'evaluate_time', 'fit_duration_seconds', 'evaluate_duration_seconds']
        
        if any(key in combined_history and combined_history[key] for key in time_keys):
            plt.figure(figsize=(10, 6))  # Mesmo tamanho dos gráficos originais
            
            # Encontrar rounds com base nas chaves disponíveis
            available_keys = [k for k in time_keys if k in combined_history and combined_history[k]]
            if not available_keys:
                logger.warning("Não foram encontradas chaves de tempo para gerar o gráfico de tempos")
                return plot_files
                
            max_len = max([len(combined_history[k]) for k in available_keys])
            rounds = list(range(1, max_len + 1))
            
            # Plotar apenas fit_time e evaluate_time
            for metric, label in [('fit_time', 'Tempo de Fit'), ('evaluate_time', 'Tempo de Evaluate')]:
                
                # Obter valores da métrica (seja com nome novo ou antigo)
                values = None
                if metric in combined_history and combined_history[metric]:
                    values = combined_history[metric]
                    
                if values:
                    # Garantir que o comprimento é compatível
                    if len(values) < max_len:
                        values = values + [None] * (max_len - len(values))
                    
                        # Filtrar pontos válidos
                    valid_points = [(r, v) for r, v in zip(rounds, values) if v is not None]
                    if valid_points:
                        valid_rounds, valid_values = zip(*valid_points)
                        metric_name = metric.split('_')[0]  # 'fit', 'evaluate', etc.
                        plt.plot(valid_rounds, valid_values, marker='o', color=colors[metric_name],
                                linestyle='-', linewidth=2, label=label)
            
            # Configurar o gráfico (fora do loop de métricas)
            plt.xlabel("Ronda", fontsize=12, fontweight='bold')
            plt.ylabel("Tempo (segundos)", fontsize=12, fontweight='bold')
            plt.title("Evolução do Tempo de Treino e Avaliação", fontsize=14, fontweight='bold')
            plt.legend(fontsize=12, framealpha=0.9)  # Legenda interna como no original
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

        # 6. GRÁFICO DE TEMPOS DE EXPLICABILIDADE (LIME, SHAP - barras)
        # Chaves para tempos de explicabilidade
        explainability_keys = ['lime_duration_seconds', 'shap_duration_seconds']
        
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
                
                # Definir cores para as barras
                bar_colors = [colors.get(name, '#333333') for name in names]
                
                bars = plt.bar(names, values, color=bar_colors)
                
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

        # 7. GRÁFICO DE COMPARAÇÃO DE ACURÁCIA - Treino, Validação e Teste
        if all(key in history for key in ['val_accuracy', 'test_accuracy']):
            plt.figure(figsize=(10, 6))
            
            # Eixo X compartilhado para todas as séries
            rounds = list(range(1, len(history['val_accuracy']) + 1))
            
            # Plotar cada série
            plt.plot(rounds, history['val_accuracy'], marker='s', color=colors['val'], 
                    linestyle='-', linewidth=2, label='Validação')
            
            if 'test_accuracy' in history and history['test_accuracy']:
                plt.plot(rounds, history['test_accuracy'], marker='^', color=colors['test'], 
                        linestyle='-', linewidth=2, label='Teste')
            
            # Configurar gráfico
            plt.xlabel("Ronda", fontsize=12, fontweight='bold')
            plt.ylabel("Acurácia", fontsize=12, fontweight='bold')
            plt.title("Comparação de Acurácia por Ronda", fontsize=14, fontweight='bold')
            plt.legend(fontsize=12, framealpha=0.9)
            plt.xticks(rounds, fontsize=10)
            plt.yticks(fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.ylim(0.5, 1.05)  # Acurácia geralmente entre 0.5 e 1
            
            plt.tight_layout()
            plot_filename = "accuracy_comparison.png"
            plot_path = base_reports / plot_filename
            plt.savefig(plot_path, dpi=120)
            plt.close()
            plot_files.append(plot_filename)
            logger.info(f"Generated plot: {plot_path}")

    except Exception as e:
        logger.error(f"Erro ao gerar gráficos de evolução: {e}", exc_info=True)
    
    return plot_files
