#!/usr/bin/env python
"""
Script para corrigir o erro de sintaxe no arquivo report_utils.py
que está faltando um bloco 'except' correspondente ao bloco 'try'.
"""

import re
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s][%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger()

def fix_syntax_error():
    """
    Corrige o erro de sintaxe no arquivo report_utils.py adicionando
    um bloco 'except' que está faltando.
    """
    file_path = 'report_utils.py'
    
    try:
        # Ler o arquivo completo
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Procurar pelo padrão de código que tem o problema
        pattern = r'# Criar uma versão formatada do arquivo de explicação LIME\s+try:.*?except Exception as e:.*?logger\.error\(f"Erro ao escrever o arquivo formatado: \{e\}", exc_info=True\)\s+'
        
        # Verificar se o padrão foi encontrado
        match = re.search(pattern, content, re.DOTALL)
        if not match:
            logger.error("Padrão de código com erro não encontrado.")
            return False
        
        # Obter o texto que precisa ser corrigido
        original_text = match.group(0)
        
        # Criar texto corrigido
        fixed_text = original_text + 'except Exception as e:\n    logger.error(f\'Erro ao criar versão formatada da explicação LIME: {e}\', exc_info=True)\n\n'
        
        # Substituir o texto original pelo corrigido
        new_content = content.replace(original_text, fixed_text)
        
        # Escrever o conteúdo modificado de volta no arquivo
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        logger.info(f"Arquivo {file_path} corrigido com sucesso.")
        return True
        
    except Exception as e:
        logger.error(f"Erro ao tentar corrigir o arquivo: {e}", exc_info=True)
        return False

if __name__ == "__main__":
    success = fix_syntax_error()
    print("Correção concluída com " + ("sucesso" if success else "falha") + ".")
