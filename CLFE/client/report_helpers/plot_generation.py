"""Helper functions for generating and saving plots for reports."""
import logging
import matplotlib.pyplot as plt
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
        val_metrics_keys = ['val_accuracy', 'val_precision', 'val_recall', 'val_f1']
        if any(key in history and history[key] for key in val_metrics_keys):
            plt.figure(figsize=(12, 7))
            rounds = list(range(1, len(history.get(val_metrics_keys[0], [])) + 1))
            if not rounds: # safety check if the primary key for rounds is empty
                for key in val_metrics_keys:
                    if key in history and history[key]:
                        rounds = list(range(1, len(history[key]) + 1))
                        break
            
            if rounds: # Proceed only if rounds could be determined
                if 'val_accuracy' in history and history['val_accuracy']:
                    plt.plot(rounds, history['val_accuracy'], marker='o', color=colors['accuracy'], 
                            linestyle='-', linewidth=2, label='Acurácia (Val)')
                if 'val_precision' in history and history['val_precision']:
                    plt.plot(rounds, history['val_precision'], marker='s', color=colors['precision'], 
                            linestyle='-', linewidth=2, label='Precisão (Val)')
                if 'val_recall' in history and history['val_recall']:
                    plt.plot(rounds, history['val_recall'], marker='^', color=colors['recall'], 
                            linestyle='-', linewidth=2, label='Recall (Val)')
                if 'val_f1' in history and history['val_f1']:
                    plt.plot(rounds, history['val_f1'], marker='d', color=colors['f1'], 
                            linestyle='-', linewidth=2, label='F1-Score (Val)')
                
                plt.xlabel("Ronda", fontsize=12, fontweight='bold')
                plt.ylabel("Métrica", fontsize=12, fontweight='bold')
                plt.title("Evolução das Métricas de Classificação (Validação)", fontsize=14, fontweight='bold')
                plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.9)
                plt.xticks(rounds, fontsize=10)
                plt.yticks(fontsize=10)
                plt.ylim(0, 1.05) # Métricas de classificação geralmente entre 0 e 1
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout(rect=[0, 0, 0.85, 1]) # Ajustar layout para legenda externa
                plot_filename = "val_classification_metrics_evolution.png"
                plot_path = base_reports / plot_filename
                plt.savefig(plot_path, dpi=120)
                plt.close()
                plot_files.append(plot_filename)
                logger.info(f"Generated plot: {plot_path}")

        # 4. GRÁFICO DE MÉTRICAS DE CLASSIFICAÇÃO (ACC, PREC, REC, F1) - Teste
        test_metrics_keys = ['test_accuracy', 'test_precision', 'test_recall', 'test_f1']
        if any(key in history and history[key] for key in test_metrics_keys):
            plt.figure(figsize=(12, 7))
            rounds = list(range(1, len(history.get(test_metrics_keys[0], [])) + 1))
            if not rounds:
                for key in test_metrics_keys:
                    if key in history and history[key]:
                        rounds = list(range(1, len(history[key]) + 1))
                        break

            if rounds: # Proceed only if rounds could be determined
                if 'test_accuracy' in history and history['test_accuracy']:
                    plt.plot(rounds, history['test_accuracy'], marker='o', color=colors['accuracy'], 
                            linestyle='--', linewidth=2, label='Acurácia (Teste)')
                if 'test_precision' in history and history['test_precision']:
                    plt.plot(rounds, history['test_precision'], marker='s', color=colors['precision'], 
                            linestyle='--', linewidth=2, label='Precisão (Teste)')
                if 'test_recall' in history and history['test_recall']:
                    plt.plot(rounds, history['test_recall'], marker='^', color=colors['recall'], 
                            linestyle='--', linewidth=2, label='Recall (Teste)')
                if 'test_f1' in history and history['test_f1']:
                    plt.plot(rounds, history['test_f1'], marker='d', color=colors['f1'], 
                            linestyle='--', linewidth=2, label='F1-Score (Teste)')
                
                plt.xlabel("Ronda", fontsize=12, fontweight='bold')
                plt.ylabel("Métrica", fontsize=12, fontweight='bold')
                plt.title("Evolução das Métricas de Classificação (Teste)", fontsize=14, fontweight='bold')
                plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.9)
                plt.xticks(rounds, fontsize=10)
                plt.yticks(fontsize=10)
                plt.ylim(0, 1.05)
                plt.grid(True, linestyle='--', alpha=0.7)
                
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                plot_filename = "test_classification_metrics_evolution.png"
                plot_path = base_reports / plot_filename
                plt.savefig(plot_path, dpi=120)
                plt.close()
                plot_files.append(plot_filename)
                logger.info(f"Generated plot: {plot_path}")

        # 5. GRÁFICO DE TEMPO POR RONDA - Comparativo fit, evaluate, LIME, SHAP
        time_keys = ['fit_time', 'evaluate_time', 'lime_time', 'shap_time']
        if any(key in history and history[key] for key in time_keys):
            plt.figure(figsize=(12, 7))
            rounds = list(range(1, len(history.get(time_keys[0], [])) + 1))
            if not rounds:
                 for key in time_keys:
                    if key in history and history[key]:
                        rounds = list(range(1, len(history[key]) + 1))
                        break           
            
            if rounds: # Proceed only if rounds could be determined
                if 'fit_time' in history and history['fit_time']:
                    plt.plot(rounds, history['fit_time'], marker='o', color=colors['fit'], 
                            linestyle='-', linewidth=2, label='Tempo de Fit')
                if 'evaluate_time' in history and history['evaluate_time']:
                    plt.plot(rounds, history['evaluate_time'], marker='s', color=colors['evaluate'], 
                            linestyle='-', linewidth=2, label='Tempo de Evaluate')
                if 'lime_time' in history and history['lime_time']:
                    plt.plot(rounds, history['lime_time'], marker='^', color=colors['lime'], 
                            linestyle='-', linewidth=2, label='Tempo de LIME')
                if 'shap_time' in history and history['shap_time']:
                    plt.plot(rounds, history['shap_time'], marker='d', color=colors['shap'], 
                            linestyle='-', linewidth=2, label='Tempo de SHAP')
                
                plt.xlabel("Ronda", fontsize=12, fontweight='bold')
                plt.ylabel("Tempo (segundos)", fontsize=12, fontweight='bold')
                plt.title("Evolução do Tempo de Execução por Ronda", fontsize=14, fontweight='bold')
                plt.legend(fontsize=12, loc='center left', bbox_to_anchor=(1, 0.5), framealpha=0.9)
                plt.xticks(rounds, fontsize=10)
                plt.yticks(fontsize=10)
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.yscale('log') # Usar escala logarítmica se os tempos variarem muito
                
                plt.tight_layout(rect=[0, 0, 0.85, 1])
                plot_filename = "time_evolution.png"
                plot_path = base_reports / plot_filename
                plt.savefig(plot_path, dpi=120)
                plt.close()
                plot_files.append(plot_filename)
                logger.info(f"Generated plot: {plot_path}")

    except Exception as e:
        logger.error(f"Erro ao gerar gráficos de evolução: {e}", exc_info=True)
    
    return plot_files
