#!/usr/bin/env python3
"""
fix_evaluate_calls.py - Corrige todas as chamadas da função evaluate() para incluir criterion
"""

def fix_evaluate_calls():
    file_path = "client/client.py"
    
    print(f"Corrigindo chamadas da função evaluate() em {file_path}...")
    
    # Ler o arquivo
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Corrigir as chamadas da função evaluate()
    fixed_lines = []
    changes_made = 0
    
    for i, line in enumerate(lines):
        # Procurar por chamadas da função evaluate sem criterion
        if 'evaluate(self.model' in line and 'criterion' not in line:
            # Verificar o padrão da chamada
            if 'device=self.device)' in line:
                # Adicionar criterion antes de device
                fixed_line = line.replace(
                    'device=self.device)',
                    'criterion=self.criterion, device=self.device)'
                )
                fixed_lines.append(fixed_line)
                changes_made += 1
                print(f"Linha {i+1} corrigida")
            elif line.strip().endswith(')'):
                # Adicionar criterion como último parâmetro
                fixed_line = line.replace(
                    ')',
                    ', criterion=self.criterion)'
                )
                fixed_lines.append(fixed_line)
                changes_made += 1
                print(f"Linha {i+1} corrigida")
            else:
                # Formato não reconhecido, manter original
                fixed_lines.append(line)
                print(f"AVISO: Linha {i+1} pode precisar correção manual: {line.strip()}")
        else:
            fixed_lines.append(line)
    
    # Salvar o arquivo corrigido
    if changes_made > 0:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(fixed_lines)
        print(f"\n{changes_made} chamadas de evaluate() corrigidas!")
    else:
        print("\nNenhuma correção necessária.")
    
    # Verificar todas as chamadas de evaluate
    print("\nVerificando todas as chamadas de evaluate()...")
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    evaluate_calls = []
    for i, line in enumerate(lines):
        if 'evaluate(' in line and 'def evaluate' not in line:
            evaluate_calls.append(f"Linha {i+1}: {line.strip()}")
    
    print(f"\nEncontradas {len(evaluate_calls)} chamadas de evaluate():")
    for call in evaluate_calls[:10]:  # Mostrar apenas as primeiras 10
        print(f"  {call}")

if __name__ == "__main__":
    fix_evaluate_calls()