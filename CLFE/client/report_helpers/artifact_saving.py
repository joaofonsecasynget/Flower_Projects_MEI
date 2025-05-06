from datetime import datetime
import torch
import logging

# Configuração do logger
logger = logging.getLogger(__name__)

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
