#!/usr/bin/env python3
"""
Script para corrigir erro de indentação no explainability_processing.py
"""

import os
import shutil
from pathlib import Path

def fix_indentation_error():
    """Corrige o erro de indentação no arquivo explainability_processing.py"""
    
    file_path = Path("client/report_helpers/explainability_processing.py")
    
    if not file_path.exists():
        print(f"Erro: Arquivo {file_path} não encontrado!")
        return False
    
    # Fazer backup
    backup_path = file_path.with_suffix('.py.backup')
    shutil.copy2(file_path, backup_path)
    print(f"Backup criado: {backup_path}")
    
    # Ler o arquivo
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Procurar e corrigir o bloco try problemático
    lines = content.split('\n')
    fixed_lines = []
    
    for i, line in enumerate(lines):
        if i == 18 and line.strip() == 'try:':  # Linha 19 (0-indexed)
            fixed_lines.append(line)
            # Garantir que as próximas linhas estejam indentadas
            continue
        elif i > 18 and i < 35:  # Área do problema
            if line.strip() and not line.startswith('except') and not line.startswith('finally'):
                # Adicionar 4 espaços de indentação se necessário
                if not line.startswith('    '):
                    fixed_lines.append('    ' + line.lstrip())
                else:
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
        else:
            fixed_lines.append(line)
    
    # Escrever o arquivo corrigido
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(fixed_lines))
    
    print(f"Arquivo {file_path} corrigido!")
    return True

if __name__ == "__main__":
    print("=== Corrigindo erro de indentação ===")
    fix_indentation_error()
    print("=== Correção concluída! ===")