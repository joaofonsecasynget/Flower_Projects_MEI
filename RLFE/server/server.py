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
    
    # Extrair perdas de treinamento (MSE)
    train_losses = [m[1].get("train_loss", 0) for m in metrics if "train_loss" in m[1]]
    avg_train_loss = sum(train_losses) / len(train_losses) if train_losses else 0
    
    # Extrair RMSE de validação
    val_rmses = [m[1].get("val_rmse", 0) for m in metrics if "val_rmse" in m[1]]
    avg_val_rmse = sum(val_rmses) / len(val_rmses) if val_rmses else 0
    
    logging.info(f"Round agregation - Avg Train Loss: {avg_train_loss:.4f}, Avg Val RMSE: {avg_val_rmse:.2f}")
    
    return {
        "train_loss": avg_train_loss,
        "val_rmse": avg_val_rmse
    }

def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """Agregação de métricas durante a avaliação."""
    if not metrics:
        return {}
    
    # Extrair e calcular a média das métricas de teste
    test_losses = [m[1].get("test_loss", 0) for m in metrics if "test_loss" in m[1]]
    avg_test_loss = sum(test_losses) / len(test_losses) if test_losses else 0
    
    test_rmses = [m[1].get("test_rmse", 0) for m in metrics if "test_rmse" in m[1]]
    avg_test_rmse = sum(test_rmses) / len(test_rmses) if test_rmses else 0
    
    logging.info(f"Evaluation agregation - Avg Test Loss: {avg_test_loss:.4f}, Avg Test RMSE: {avg_test_rmse:.2f}")
    
    return {
        "test_loss": avg_test_loss,
        "test_rmse": avg_test_rmse
    }

# Definir parâmetros do servidor
server_address = "0.0.0.0:9091"
NUM_ROUNDS = 5  # Pode ser ajustado conforme necessário

# Configuração do servidor
config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS)

# Estratégia de federação
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Usar todos os clientes disponíveis para treinamento
    fraction_evaluate=1.0,  # Usar todos os clientes disponíveis para avaliação
    min_fit_clients=2,  # Número mínimo de clientes para treinamento
    min_evaluate_clients=2,  # Número mínimo de clientes para avaliação
    min_available_clients=2,  # Espera pelo menos este número de clientes antes de iniciar
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
)

if __name__ == "__main__":
    logging.info(f"Iniciando servidor Flower em {server_address} com {NUM_ROUNDS} rounds...")
    fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )
