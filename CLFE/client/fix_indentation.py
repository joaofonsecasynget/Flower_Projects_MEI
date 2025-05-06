#!/usr/bin/env python
"""
Script para corrigir o erro de indentação no arquivo report_utils.py
"""

import re
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

def fix_indentation_error():
    """
    Corrige o erro de indentação no arquivo report_utils.py
    """
    file_path = 'report_utils.py'
    
    try:
        # Ler o arquivo linha por linha
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Encontrar a linha com o problema
        for i, line in enumerate(lines):
            if "except Exception as e:" in line and "Erro ao criar versão formatada da explicação LIME" in line:
                # Encontrou a linha problemática
                # Separar em duas linhas com indentação correta
                indent = ' ' * (len(line) - len(line.lstrip()))
                logger_line = f"{indent}    logger.error(f'Erro ao criar versão formatada da explicação LIME: {{e}}', exc_info=True)\n"
                lines[i] = f"{indent}except Exception as e:\n{logger_line}"
                logger.info(f"Corrigida a indentação na linha {i+1}")
                break
        
        # Salvar o arquivo corrigido
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)
        
        logger.info(f"Arquivo {file_path} corrigido com sucesso.")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao tentar corrigir o arquivo: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = fix_indentation_error()
    print("Correção de indentação concluída com " + ("sucesso" if success else "falha") + ".")
