# Projeto de Aprendizagem Federada e Explicabilidade para Ambientes IoT

Este projeto implementa estratégias avançadas de aprendizagem federada (Federated Learning - FL) com foco em ambientes distribuídos e seguros, utilizando um novo dataset de IoT. O objetivo é combinar desempenho preditivo com explicabilidade, explorando modelos lineares e árvores de decisão, e integrando técnicas como LIME e SHAP.

## Título da Dissertação
"Advanced Federated Learning Strategies: A Multi-Model Approach for Distributed and Secure Environments"

## Estrutura do Projeto
- **CLFE/**: Nova implementação principal, totalmente containerizada com Docker e Docker Compose.
- **DatasetIOT/**: Contém o dataset IoT real utilizado (`transformed_dataset_imeisv_8642840401612300.csv`).
- **reports/** e **results/**: Volumes persistentes para outputs e artefactos de cada cliente.

## Abordagens
- **ADF**: Árvore de Decisão Federada (DecisionTreeRegressor, scikit-learn)
- **CLFE**: Classificação Linear Federada Explicável (PyTorch, LIME/SHAP)

## Principais Características
- Orquestração completa via Docker Compose, escalando automaticamente o número de clientes.
- Cada cliente executa em container isolado, com partição estratificada do dataset (mantendo distribuição da variável target).
- Outputs organizados por cliente: métricas, histórico, plots, imagens de explicabilidade e relatório HTML consolidado.
- Particionamento estratificado dos dados preservando a distribuição do target em cada cliente
- Código modular e organizado com funções auxiliares para explicabilidade e geração de relatórios
- Relatórios de explicabilidade interativos com formatação inteligente e acesso a todos os valores da instância explicada

## Instruções de Execução

### Abordagem CLFE (Classificação Linear Federada Explicável)

Esta abordagem utiliza Docker e Docker Compose para orquestrar um servidor Flower e múltiplos clientes que treinam um modelo de classificação linear (PyTorch) de forma federada e geram explicações (LIME/SHAP) na ronda final.

**Passos:**

1.  **Navegar para a pasta CLFE:**
    Abra um terminal e entre no diretório da implementação CLFE.
    ```sh
    cd CLFE
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

*   Os outputs (relatórios, modelos, gráficos) serão guardados nas pastas `CLFE/reports/client_X` e `CLFE/results/client_X`.
*   O volume `DatasetIOT/` é montado como read-only nos contentores.
*   Certifique-se de que o Docker Desktop (ou Docker Engine) está em execução antes de correr os comandos `docker compose`.

### Abordagem ADF (Árvore de Decisão Federada)

*(Instruções detalhadas a adicionar)*

## Execução Local de Cliente CLFE

```bash
python client.py --cid=1 --num_clients=4 --num_total_clients=4 --dataset_path=../DatasetIOT/transformed_dataset_imeisv_8642840401612300.csv
```

## Argumentos do Cliente CLFE
- `--cid`: ID do cliente (1-indexed)
- `--num_clients`: número de clientes a executar neste teste
- `--num_total_clients`: número total de clientes para particionamento dos dados
- `--dataset_path`: caminho para o dataset CSV
- `--seed`: seed para reprodutibilidade

O dataset é particionado usando `StratifiedKFold` para preservar a proporção da variável target (attack) em cada cliente, tornando o treinamento mais representativo e as explicações mais robustas.

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
    - `reports/client_X/shap_feature_type_importance.png` (Importância por tipo de feature)
    - `reports/client_X/shap_temporal_trends.png` (Tendências temporais)
    - `reports/client_X/shap_temporal_heatmap.png` (Mapa de calor temporal)
    - `reports/client_X/shap_temporal_index_importance.png` (Importância por índice temporal)
    - `reports/client_X/shap_timestamp_features_importance.png` (Importância das features derivadas de timestamp)
- **Relatório HTML Consolidado:** `reports/client_X/final_report.html` (métricas por ronda, tempos, gráficos, LIME Top 10, SHAP)
- **Ferramenta de Explicabilidade Interativa:** `explain_instance.py` (permite selecionar e explicar qualquer instância individual)

## Estado Atual
- Conversão de RLFE (Regressão Linear) para CLFE (Classificação Linear) concluída com sucesso
- Nova estrutura CLFE funcional e estável, utilizando o dataset IoT para detecção de ataques
- Correção da última ronda para garantir métricas completas em todas as rondas
- Interface de relatório otimizada com tabelas divididas por tipo de métrica (treino/validação, teste, tempos)
- Particionamento estratificado para garantir distribuição equilibrada do target em cada cliente
- Excelente desempenho do modelo (Acurácia ~99%, Precision ~98%, Recall ~97%, F1-score ~98%)
- Explicabilidade (LIME/SHAP) gerando visualizações categorizadas por tipo de feature
- Relatórios HTML finais detalhados com múltiplos gráficos e visualizações
- Implementação do sistema de metadados para rastreabilidade completa das features
- **NOVO:** Correção de gráficos para mostrar o número correto de rondas (5 em vez de 10)
- **NOVO:** Interface de relatório melhorada com apresentação lado a lado de LIME/SHAP
- **NOVO:** Reorganização lógica dos gráficos de explicabilidade e evolução de métricas

## Próximos Passos
- Executar simulações com diferentes números de clientes/rondas para popular a `COMPARACAO_FORMAL.md`
- Comparar formalmente os resultados da CLFE com a abordagem ADF no contexto do dataset IoT
- Investigar razões para o excepcional desempenho do modelo linear na detecção de ataques
- Aprofundar a análise das features temporais que demonstraram alta importância
- Resolver a incompatibilidade entre features do modelo e do dataset original
- Utilizar os artefactos gerados para finalizar a escrita da dissertação