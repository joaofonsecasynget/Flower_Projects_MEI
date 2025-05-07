#!/usr/bin/env python
from pathlib import Path
import json
import sys
import os

# Adicionar o diretório atual ao path para que possa importar módulos do cliente
sys.path.append(os.path.join(os.path.dirname(__file__), 'client'))

# Importar o módulo de geração de relatório HTML
from client.report_helpers.html_generation import generate_html_report

# Configurar o diretório base dos relatórios
base_reports = Path('client/reports/client_1')

# Carregar o histórico de métricas
with open(base_reports/'metrics_history.json') as f:
    history = json.load(f)

# Obter a lista de arquivos de plot
plot_files = [p.name for p in base_reports.glob('*.png')]

# Gerar o relatório HTML
generate_html_report(
    history, 
    plot_files, 
    base_reports, 
    1, 
    'DatasetIOT/transformed_dataset_imeisv_8642840401612300.csv', 
    42, 
    5, 
    'server:9091'
)

print(f"Relatório HTML atualizado com sucesso em {base_reports}/detailed_report_client_1.html")
