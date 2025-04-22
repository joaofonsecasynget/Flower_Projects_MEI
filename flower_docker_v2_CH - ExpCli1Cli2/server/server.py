import flwr as fl
from typing import List, Tuple, Dict, Optional
from flwr.common import Metrics

def fit_metrics_aggregation_fn(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    mse_values = [m[1].get("mse", 0) for m in metrics if "mse" in m[1]]
    aggregated_mse = sum(mse_values) / len(mse_values) if mse_values else 0
    print(f"[Server] Agregação de métricas (treino) - MSE Médio: {aggregated_mse:.4f}")
    return {"mse": aggregated_mse}

def evaluate_metrics_aggregation_fn(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    mse_values = [m[1].get("mse", 0) for m in metrics if "mse" in m[1]]
    aggregated_mse = sum(mse_values) / len(mse_values) if mse_values else 0
    print(f"[Server] Agregação de métricas (avaliação) - MSE Médio: {aggregated_mse:.4f}")
    return {"mse": aggregated_mse}

server_address = "0.0.0.0:9091"
config = fl.server.ServerConfig(num_rounds=5)

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=2,
    min_evaluate_clients=2,
    min_available_clients=2,
    fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
    evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
)

if __name__ == "__main__":
    print(f"[Server] Iniciando servidor Flower em {server_address} com {config.num_rounds} rounds...")
    fl.server.start_server(
        server_address=server_address,
        config=config,
        strategy=strategy,
    )
