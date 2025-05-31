import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][SERVER][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Agregação de métricas durante o treinamento."""
    # Extrair e calcular a média das métricas relevantes
    if not metrics:
        return {}
    
    # Extrair perdas de treinamento (BCE)
    train_losses = [m[1].get("train_loss", 0) for m in metrics if "train_loss" in m[1]]
    avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0
    
    # Extrair métricas de validação
    val_accuracies = [m[1].get("val_accuracy", 0) for m in metrics if "val_accuracy" in m[1]]
    avg_val_accuracy = sum(val_accuracies) / len(val_accuracies) if val_accuracies else 0
    
    val_precisions = [m[1].get("val_precision", 0) for m in metrics if "val_precision" in m[1]]
    avg_val_precision = sum(val_precisions) / len(val_precisions) if val_precisions else 0
    
    val_recalls = [m[1].get("val_recall", 0) for m in metrics if "val_recall" in m[1]]
    avg_val_recall = sum(val_recalls) / len(val_recalls) if val_recalls else 0
    
    val_f1s = [m[1].get("val_f1", 0) for m in metrics if "val_f1" in m[1]]
    avg_val_f1 = sum(val_f1s) / len(val_f1s) if val_f1s else 0
    
    # Para compatibilidade, ainda mantemos RMSE
    val_rmses = [m[1].get("val_rmse", 0) for m in metrics if "val_rmse" in m[1]]
    avg_val_rmse = sum(val_rmses) / len(val_rmses) if val_rmses else 0
    
    logging.info(f"Round agregation - Avg Train Loss: {avg_train_loss:.4f}")
    logging.info(f"Metrics - Acc: {avg_val_accuracy:.4f}, Prec: {avg_val_precision:.4f}, Recall: {avg_val_recall:.4f}, F1: {avg_val_f1:.4f}")
    
    return {
        "train_loss": avg_train_loss,
        "val_rmse": avg_val_rmse,
        "val_accuracy": avg_val_accuracy,
        "val_precision": avg_val_precision,
        "val_recall": avg_val_recall,
        "val_f1": avg_val_f1
    }

def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Agregação de métricas durante a avaliação."""
    if not metrics:
        return {}
    
    # Extrair e calcular a média das métricas de teste
    test_losses = [m[1].get("test_loss", 0) for m in metrics if "test_loss" in m[1]]
    avg_test_loss = sum(test_losses) / len(test_losses) if test_losses else 0
    
    # Para compatibilidade, ainda mantemos RMSE
    test_rmses = [m[1].get("test_rmse", 0) for m in metrics if "test_rmse" in m[1]]
    avg_test_rmse = sum(test_rmses) / len(test_rmses) if test_rmses else 0
    
    # Métricas de classificação
    test_accuracies = [m[1].get("test_accuracy", 0) for m in metrics if "test_accuracy" in m[1]]
    avg_test_accuracy = sum(test_accuracies) / len(test_accuracies) if test_accuracies else 0
    
    test_precisions = [m[1].get("test_precision", 0) for m in metrics if "test_precision" in m[1]]
    avg_test_precision = sum(test_precisions) / len(test_precisions) if test_precisions else 0
    
    test_recalls = [m[1].get("test_recall", 0) for m in metrics if "test_recall" in m[1]]
    avg_test_recall = sum(test_recalls) / len(test_recalls) if test_recalls else 0
    
    test_f1s = [m[1].get("test_f1", 0) for m in metrics if "test_f1" in m[1]]
    avg_test_f1 = sum(test_f1s) / len(test_f1s) if test_f1s else 0
    
    logging.info(f"Evaluation - Avg Test Loss: {avg_test_loss:.4f}")
    logging.info(f"Metrics - Acc: {avg_test_accuracy:.4f}, Prec: {avg_test_precision:.4f}, Recall: {avg_test_recall:.4f}, F1: {avg_test_f1:.4f}")
    
    return {
        "test_loss": avg_test_loss,
        "test_rmse": avg_test_rmse,
        "test_accuracy": avg_test_accuracy,
        "test_precision": avg_test_precision,
        "test_recall": avg_test_recall,
        "test_f1": avg_test_f1
    }

# Estratégia de federação com configuração atualizada
from flwr.server.strategy import FedAvg

class RLFEFedAvg(FedAvg):
    def __init__(self, *args, num_rounds=5, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_rounds = num_rounds
        
    def configure_fit(self, server_round, parameters, client_manager, **kwargs):
        # Chama a implementação base para obter as instruções padrão
        ins = super().configure_fit(server_round, parameters, client_manager, **kwargs)
        # Injeta round_number e num_rounds na config de cada cliente
        for i in range(len(ins)):  # ins é uma lista de tuplos (ClientProxy, FitIns)
            if hasattr(ins[i][1], 'config') and isinstance(ins[i][1].config, dict):
                ins[i][1].config['round_number'] = server_round
                ins[i][1].config['num_rounds'] = self.num_rounds if hasattr(self, 'num_rounds') else 0
        return ins
        
    def configure_evaluate(self, server_round, parameters, client_manager, **kwargs):
        eval_ins = super().configure_evaluate(server_round, parameters, client_manager, **kwargs)
        for ins in eval_ins:
            if hasattr(ins[1], 'config') and isinstance(ins[1].config, dict):
                ins[1].config['round_number'] = server_round
                ins[1].config['num_rounds'] = self.num_rounds if hasattr(self, 'num_rounds') else 0
        return eval_ins

if __name__ == "__main__":
    # Definir parâmetros do servidor
    server_address = "0.0.0.0:9091"
    NUM_ROUNDS = 5
    
    logging.info(f"Iniciando servidor Flower em {server_address} com {NUM_ROUNDS} rounds...")
    
    # Configuração do servidor
    config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS)
    
    # Estratégia personalizada
    strategy = RLFEFedAvg(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=2,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        num_rounds=NUM_ROUNDS
    )
    
    # Usar método atualizado (compatível mas ainda funcional)
    try:
        fl.server.start_server(
            server_address=server_address,
            config=config,
            strategy=strategy,
        )
    except Exception as e:
        logging.error(f"Erro ao iniciar servidor com método legacy: {e}")
        logging.info("Para versões mais recentes do Flower, use: flower-superlink --insecure")