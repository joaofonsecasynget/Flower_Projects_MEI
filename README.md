# Projeto de Aprendizagem Federada e Explicabilidade para Ambientes IoT

Este projeto implementa estratégias avançadas de aprendizagem federada (Federated Learning - FL) com foco em ambientes distribuídos e seguros, utilizando um novo dataset de IoT. O objetivo é combinar desempenho preditivo com explicabilidade, explorando modelos lineares e árvores de decisão, e integrando técnicas como LIME e SHAP.

## Título da Dissertação
"Advanced Federated Learning Strategies: A Multi-Model Approach for Distributed and Secure Environments"

## Estrutura do Projeto
- **RLFE/**: Nova implementação principal, totalmente containerizada com Docker e Docker Compose.
- **DatasetIOT/**: Contém o dataset IoT real utilizado (`transformed_dataset_imeisv_8642840401612300.csv`).
- **reports/** e **results/**: Volumes persistentes para outputs e artefactos de cada cliente.

## Abordagens
- **ADF**: Árvore de Decisão Federada (DecisionTreeRegressor, scikit-learn)
- **RLFE**: Regressão Linear Federada Explicável (PyTorch, LIME/SHAP)

## Principais Características
- Orquestração completa via Docker Compose, escalando automaticamente o número de clientes.
- Cada cliente executa em container isolado, com partição distinta do dataset (seed fixa para reprodutibilidade).
- Outputs organizados por cliente: métricas, histórico, plots, imagens de explicabilidade e relatório HTML consolidado.
- Explicabilidade gerada apenas após a última ronda federada, garantindo que os artefactos finais refletem o treino completo.
- Healthcheck garante que clientes só arrancam após o servidor estar disponível.

## Execução Federada (Docker)

1. Entrar na pasta RLFE:
   ```sh
   cd RLFE
   ```
2. Gerar o docker-compose:
   ```sh
   python generate_compose.py <NUM_CLIENTES> <NUM_ROUNDS>
   ```
3. Subir os serviços federados:
   ```sh
   docker compose -f docker-compose.generated.yml up --build --detach
   ```
4. Monitorizar logs e outputs:
   ```sh
   docker compose -f docker-compose.generated.yml logs --tail=100
   ```

- O volume `DatasetIOT/` é montado como read-only; `reports/` e `results/` são persistentes.
- Para personalizar volumes, nomes de container ou adicionar serviços extra (ex: servidor FL), basta ajustar o template no script `generate_compose.py`.

## Execução Local de Cliente RLFE

```bash
python client.py --cid=1 --num_clients=4 --num_total_clients=4 --dataset_path=../DatasetIOT/transformed_dataset_imeisv_8642840401612300.csv
```

## Argumentos do Cliente RLFE
- `--cid`: ID do cliente (1-indexed)
- `--num_clients`: número de clientes a executar neste teste
- `--num_total_clients`: número total de clientes para particionamento dos dados
- `--dataset_path`: caminho para o dataset CSV
- `--seed`: seed para reprodutibilidade

O dataset é sempre dividido por `num_total_clients` e cada cliente recebe apenas a sua fração, tornando o teste mais realista e eficiente mesmo com poucos clientes em execução.

## Outputs Gerados por Cliente (ao final do ciclo federado)
- **Modelo treinado:** `results/client_X/model_client_X.pt`
- **Histórico de métricas:** `reports/client_X/metrics_history.json`
- **Plots:** `reports/client_X/loss_evolution.png`, `reports/client_X/rmse_evolution.png`
- **Explicabilidade:** `reports/client_X/lime_final.png`, `reports/client_X/shap_final.png`
- **Relatório HTML:** `reports/client_X/final_report.html` (consolidando métricas, gráficos e imagens)

## Estado Atual
- Nova estrutura RLFE funcional, substituindo abordagens anteriores.
- Explicabilidade, outputs e relatórios finais só gerados na última ronda.
- Testes e validação em curso com múltiplos clientes e dataset IoT.
- Documentação e exemplos de outputs serão incrementados após validação.

## Próximos Passos
- Comparação formal dos resultados entre ADF e RLFE (tabela de métricas, análise de explicabilidade, discussão de limitações).
- Desenvolver protocolo de testes e critérios para comparação robusta.
- Incrementar documentação com exemplos reais dos artefactos gerados.