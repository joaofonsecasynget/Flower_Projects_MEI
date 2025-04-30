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
- **Explicabilidade (LIME/SHAP):**
    - Gerada **apenas após a última ronda federada**, refletindo o modelo final do cliente.
    - **LIME:** Imagem no relatório principal (`lime_final.png`) mostra o **Top 10 features**. Relatório HTML separado (`lime_final.html`) e ficheiro de texto (`lime_explanation.txt`) contêm a explicação completa.
    - **SHAP:** Imagem (`shap_final.png`) e valores (`shap_values.npy`) gerados.
- **Relatórios Detalhados:**
    - Métricas por ronda (incluindo tempos de `fit` e `evaluate`) apresentadas em tabela.
    - Tempos de execução medidos com `time.perf_counter` para maior precisão.
    - Gráficos de evolução para todas as métricas principais.
    - Plots de métricas organizados em grelha no HTML.
- Healthcheck garante que clientes só arrancam após o servidor estar disponível.

## Instruções de Execução

### Abordagem RLFE (Regressão Linear Federada Explicável)

Esta abordagem utiliza Docker e Docker Compose para orquestrar um servidor Flower e múltiplos clientes que treinam um modelo de regressão linear (PyTorch) de forma federada e geram explicações (LIME/SHAP) na ronda final.

**Passos:**

1.  **Navegar para a pasta RLFE:**
    Abra um terminal e entre no diretório da implementação RLFE.
    ```sh
    cd RLFE
    ```

2.  **Gerar o Ficheiro Docker Compose:**
    Execute o script Python para gerar o ficheiro `docker-compose.generated.yml`. Este script configura os serviços do servidor e o número desejado de clientes e rondas.
    ```sh
    python generate_compose.py <NUM_CLIENTES> <NUM_ROUNDS>
    ```
    *   `<NUM_CLIENTES>`: Substitua pelo número de clientes que deseja simular (e.g., `2`).
    *   `<NUM_ROUNDS>`: Substitua pelo número de rondas de treino federado (e.g., `5`).
    *   **Exemplo:** `python generate_compose.py 2 5` irá configurar 1 servidor e 2 clientes para 5 rondas.

3.  **Iniciar os Serviços Federados:**
    Utilize o Docker Compose para construir as imagens (se necessário) e iniciar os contentores do servidor e dos clientes em modo 'detached' (background).
    ```sh
    docker compose -f docker-compose.generated.yml up --build --detach
    ```
    *   `-f docker-compose.generated.yml`: Especifica o ficheiro de configuração gerado.
    *   `up`: Comando para criar e iniciar os contentores.
    *   `--build`: Força a reconstrução das imagens Docker se houver alterações no código ou Dockerfile desde a última build. É recomendado usar na primeira vez ou após modificações.
    *   `--detach`: Executa os contentores em segundo plano, libertando o terminal.

4.  **Monitorizar os Logs (Opcional):**
    Pode acompanhar o progresso do treino e a comunicação entre clientes e servidor visualizando os logs.
    ```sh
    docker compose -f docker-compose.generated.yml logs --tail=100 -f
    ```
    *   `logs`: Comando para mostrar os logs dos serviços.
    *   `--tail=100`: Mostra as últimas 100 linhas de log de cada serviço.
    *   `-f` (follow): Continua a mostrar novos logs à medida que são gerados. Pressione `Ctrl+C` para parar de seguir.

5.  **Parar os Serviços:**
    Quando o treino federado terminar (ou se desejar interromper), pode parar e remover os contentores e a rede associada.
    ```sh
    docker compose -f docker-compose.generated.yml down --remove-orphans
    ```
    *   `down`: Para e remove os contentores, redes e volumes (se não forem externos) definidos no ficheiro compose.
    *   `--remove-orphans`: Remove contentores de serviços que não estão mais definidos no ficheiro compose (útil para limpeza).

**Notas Importantes:**

*   Os outputs (relatórios, modelos, gráficos) serão guardados nas pastas `RLFE/reports/client_X` e `RLFE/results/client_X`.
*   O volume `DatasetIOT/` é montado como read-only nos contentores.
*   Certifique-se de que o Docker Desktop (ou Docker Engine) está em execução antes de correr os comandos `docker compose`.

### Abordagem ADF (Árvore de Decisão Federada)

*(Instruções detalhadas a adicionar)*

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
- **Dataset de treino (para LIME/SHAP):** `reports/client_X/X_train.npy`
- **Plots de Evolução:** `reports/client_X/<metrica>_evolution.png` (e.g., `train_loss_evolution.png`, `val_rmse_evolution.png`, etc.)
- **Explicabilidade LIME:**
    - `reports/client_X/lime_final.png` (Top 10 Features)
    - `reports/client_X/lime_final.html` (Explicação Completa)
    - `reports/client_X/lime_explanation.txt` (Explicação em Texto)
- **Explicabilidade SHAP:**
    - `reports/client_X/shap_final.png`
    - `reports/client_X/shap_values.npy`
- **Relatório HTML Consolidado:** `reports/client_X/final_report.html` (métricas por ronda, tempos, gráficos, LIME Top 10, SHAP)

## Estado Atual
- Nova estrutura RLFE funcional e estável, utilizando o dataset IoT.
- Corrigidos erros anteriores (matplotlib, TypeError LIME, falta de X_train.npy).
- Explicabilidade (LIME/SHAP) e relatórios finais detalhados (com tempos precisos, métricas por ronda, múltiplos plots) gerados apenas na última ronda.
- Testes confirmam o comportamento esperado: outputs corretos, explicabilidade apenas no final, tempos registados.
- Documentação (este README, ESTADOATUAL.md) atualizada para refletir as últimas modificações.

## Próximos Passos
- Executar simulações com diferentes números de clientes/rondas para popular a `COMPARACAO_FORMAL.md` com resultados do dataset IoT.
- Comparar formalmente os resultados da RLFE com a abordagem ADF (se aplicável ao dataset IoT) ou outras baselines.
- Utilizar os artefactos gerados (relatórios, gráficos, explicações) para a escrita da dissertação.
- Refinar a análise da explicabilidade LIME/SHAP no contexto do problema IoT.