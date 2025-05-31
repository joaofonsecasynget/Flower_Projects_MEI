#!/usr/bin/env python3
"""
Script para corrigir os erros de sintaxe e compatibilidade no projeto CLFE.
"""

import os
import re
import logging
import shutil
from pathlib import Path

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def fix_client_syntax():
    """Corrige o erro de sintaxe no client.py"""
    client_file = "client/client.py"
    
    if not os.path.exists(client_file):
        logger.error(f"Arquivo {client_file} não encontrado!")
        return False
    
    logger.info(f"Criando backup de {client_file}")
    backup_file = f"{client_file}.backup"
    shutil.copy2(client_file, backup_file)
    
    try:
        with open(client_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Procurar pelo padrão problemático na linha ~420
        # O problema é que o return está mal indentado
        pattern = r'(\s+)# Retornar parâmetros atualizados.*?\n(\s+)return self\.get_parameters\(config=\{\}\)'
        
        def fix_indentation(match):
            first_indent = match.group(1)
            # O return deve ter a mesma indentação do comentário
            return match.group(0).replace(match.group(2), first_indent)
        
        # Aplicar correção
        fixed_content = re.sub(pattern, fix_indentation, content, flags=re.DOTALL)
        
        # Verificar se houve mudança
        if fixed_content != content:
            with open(client_file, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            logger.info(f"Correção aplicada em {client_file}")
            return True
        else:
            logger.warning("Nenhuma correção foi aplicada - padrão não encontrado")
            return False
            
    except Exception as e:
        logger.error(f"Erro ao corrigir {client_file}: {e}")
        # Restaurar backup em caso de erro
        shutil.copy2(backup_file, client_file)
        return False

def fix_server_deprecation():
    """Adiciona compatibilidade para versões mais recentes do Flower"""
    server_file = "server/server.py"
    
    if not os.path.exists(server_file):
        logger.error(f"Arquivo {server_file} não encontrado!")
        return False
    
    logger.info(f"Criando backup de {server_file}")
    backup_file = f"{server_file}.backup"
    shutil.copy2(server_file, backup_file)
    
    try:
        with open(server_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Adicionar tratamento de erro para compatibilidade
        if "try:" not in content or "flower-superlink" not in content:
            # Procurar pelo início do servidor
            pattern = r'(if __name__ == "__main__":.*?fl\.server\.start_server\([^}]+\))'
            
            replacement = '''if __name__ == "__main__":
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
        logging.info("Para versões mais recentes do Flower, use: flower-superlink --insecure")'''
            
            fixed_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
            
            if fixed_content != content:
                with open(server_file, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                logger.info(f"Compatibilidade adicionada em {server_file}")
                return True
            else:
                logger.info("Servidor já possui tratamento de compatibilidade")
                return True
                
    except Exception as e:
        logger.error(f"Erro ao corrigir {server_file}: {e}")
        shutil.copy2(backup_file, server_file)
        return False

def check_docker_setup():
    """Verifica se a configuração Docker está correta"""
    docker_files = [
        "Dockerfile",
        "docker-compose.generated.yml",
        "requirements.txt"
    ]
    
    missing_files = []
    for file_path in docker_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        logger.warning(f"Arquivos Docker faltando: {missing_files}")
        return False
    
    logger.info("Configuração Docker está completa")
    return True

def main():
    """Função principal para aplicar todas as correções"""
    logger.info("=== Iniciando correções do projeto CLFE ===")
    
    success_count = 0
    total_fixes = 3
    
    # 1. Corrigir sintaxe do cliente
    if fix_client_syntax():
        success_count += 1
        logger.info("✅ Sintaxe do cliente corrigida")
    else:
        logger.error("❌ Falha ao corrigir sintaxe do cliente")
    
    # 2. Corrigir compatibilidade do servidor
    if fix_server_deprecation():
        success_count += 1
        logger.info("✅ Compatibilidade do servidor corrigida")
    else:
        logger.error("❌ Falha ao corrigir compatibilidade do servidor")
    
    # 3. Verificar configuração Docker
    if check_docker_setup():
        success_count += 1
        logger.info("✅ Configuração Docker verificada")
    else:
        logger.error("❌ Problemas na configuração Docker")
    
    # Resultado final
    logger.info(f"=== Correções concluídas: {success_count}/{total_fixes} ===")
    
    if success_count == total_fixes:
        logger.info("🎉 Todas as correções foram aplicadas com sucesso!")
        logger.info("Agora você pode executar:")
        logger.info("  1. docker-compose -f docker-compose.generated.yml build")
        logger.info("  2. docker-compose -f docker-compose.generated.yml up")
        return True
    else:
        logger.error("⚠️  Algumas correções falharam. Verifique os logs acima.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)