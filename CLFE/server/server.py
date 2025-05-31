#!/usr/bin/env python3
"""
Correção manual específica para o servidor
"""
import os
import shutil

def fix_server_manual():
    """Corrige o servidor manualmente"""
    server_file = "server/server.py"
    
    if not os.path.exists(server_file):
        print(f"❌ Arquivo {server_file} não encontrado!")
        return False
    
    # Fazer backup
    backup_file = f"{server_file}.backup"
    shutil.copy2(server_file, backup_file)
    print(f"📋 Backup criado: {backup_file}")
    
    try:
        with open(server_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Verificar se já tem tratamento de erro
        if "try:" in content and "flower-superlink" in content:
            print("✅ Servidor já possui tratamento de compatibilidade")
            return True
        
        # Procurar pelo final do arquivo onde está o start_server
        lines = content.split('\n')
        new_lines = []
        in_main_block = False
        
        for line in lines:
            if 'if __name__ == "__main__":' in line:
                in_main_block = True
                new_lines.append(line)
                new_lines.append('    # Definir parâmetros do servidor')
                new_lines.append('    server_address = "0.0.0.0:9091"')
                new_lines.append('    NUM_ROUNDS = 5')
                new_lines.append('    ')
                new_lines.append('    logging.info(f"Iniciando servidor Flower em {server_address} com {NUM_ROUNDS} rounds...")')
                new_lines.append('    ')
                new_lines.append('    # Configuração do servidor')
                new_lines.append('    config = fl.server.ServerConfig(num_rounds=NUM_ROUNDS)')
                new_lines.append('    ')
                new_lines.append('    # Estratégia personalizada')
                new_lines.append('    strategy = RLFEFedAvg(')
                new_lines.append('        fraction_fit=1.0,')
                new_lines.append('        fraction_evaluate=1.0,')
                new_lines.append('        min_fit_clients=2,')
                new_lines.append('        min_evaluate_clients=2,')
                new_lines.append('        min_available_clients=2,')
                new_lines.append('        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,')
                new_lines.append('        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,')
                new_lines.append('        num_rounds=NUM_ROUNDS')
                new_lines.append('    )')
                new_lines.append('    ')
                new_lines.append('    # Usar método com tratamento de erro para compatibilidade')
                new_lines.append('    try:')
                new_lines.append('        fl.server.start_server(')
                new_lines.append('            server_address=server_address,')
                new_lines.append('            config=config,')
                new_lines.append('            strategy=strategy,')
                new_lines.append('        )')
                new_lines.append('    except Exception as e:')
                new_lines.append('        logging.error(f"Erro ao iniciar servidor com método legacy: {e}")')
                new_lines.append('        logging.info("Para versões mais recentes do Flower, use: flower-superlink --insecure")')
                
                # Pular as linhas antigas do main
                continue
            elif in_main_block and (line.strip().startswith('fl.server.start_server') or 
                                  line.strip().startswith('server_address') or
                                  line.strip().startswith('NUM_ROUNDS') or
                                  line.strip().startswith('config') or
                                  line.strip().startswith('strategy') or
                                  line.strip().startswith('logging.info')):
                # Pular linhas antigas que já foram substituídas
                continue
            else:
                new_lines.append(line)
        
        # Escrever arquivo corrigido
        with open(server_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(new_lines))
        
        print("✅ Servidor corrigido com sucesso!")
        return True
        
    except Exception as e:
        print(f"❌ Erro ao corrigir servidor: {e}")
        # Restaurar backup
        shutil.copy2(backup_file, server_file)
        return False

if __name__ == "__main__":
    print("=== Correção Manual do Servidor ===")
    success = fix_server_manual()
    
    if success:
        print("\n🎉 Correção do servidor aplicada!")
        print("Agora o sistema está pronto para executar:")
        print("  docker-compose -f docker-compose.generated.yml build")
        print("  docker-compose -f docker-compose.generated.yml up")
    else:
        print("\n❌ Falha na correção do servidor")
    
    exit(0 if success else 1)