# Estado Atual do Projeto e Tarefas Futuras

Este documento acompanha o progresso do projeto de Aprendizagem Federada e Explicabilidade para Previsão de Preços Imobiliários, no âmbito de uma dissertação de mestrado.

## Legenda

- [ ] Tarefa por fazer
- [/] Tarefa em andamento
- [x] Tarefa concluída
- [?] Tarefa concluída, mas a necessitar de verificação/teste

## Tarefas Gerais / Próximos Passos Principais

- [/] **Verificação das Implementações Atuais:**
    - [x] 1.1: Testar execução `docker-compose up` para `DecisionTreeRegressor`.
    - [x] 1.2: Verificar outputs (logs, métricas, relatórios) da `DecisionTreeRegressor` vs `resumoDTR.md`. (Logs e métricas OK. Gráficos de evolução OK, mas notar plot de Soma Importância pouco útil e clarificar cálculo/apresentação da Similaridade Estrutural final vs evolução no gráfico).
    - [x] 1.3: Testar execução `docker-compose up` para `LinearRegression`.
    - [x] 1.4: Verificar outputs (logs, métricas, relatórios) da `LinearRegression` vs `resumo_ExpC1C2.md`. (Logs e métricas (RMSE, tempo) correspondem bem ao resumo. Confirmado RMSE elevado e convergência lenta. Relatórios LIME/SHAP gerados como esperado).
    - [x] 1.5: Analisar consistência do pré-processamento (`utils.py` em `client_1`/`client_2` de ambas as abordagens). (Linear Regression: OK, `client_X.py` usa `load_datasets` idêntica. Decision Tree: INCONSISTENTE, `client_2.py` usa `load_data` diferente de `load_datasets` em `client_1.py` - tratar NaN, encoding, particionamento).
- [ ] **Transição para Novo Dataset:**
    - [ ] Definir/Obter o novo dataset alvo para a dissertação.
    - [ ] Adaptar o código de carregamento e pré-processamento para o novo dataset.
- [ ] **Comparação Formal das Abordagens:**
    - [ ] Definir métricas e critérios claros para a comparação (desempenho - e.g., RMSE no novo dataset, estabilidade entre rondas/clientes, qualidade/robustez da explicabilidade, tempo de execução, etc.).
    - [ ] Desenvolver um protocolo de teste padronizado para aplicar a ambas as abordagens com o novo dataset.
    - [ ] Executar as experiências comparativas.
- [ ] **Desenvolvimento da Dissertação:**
    - [ ] Estruturar o documento da dissertação.
    - [/] Escrever secções relevantes à medida que o trabalho progride (e.g., Introdução, Revisão da Literatura, Metodologia inicial baseada nas implementações atuais).

## Abordagem 1: Decision Tree Regressor (`flower_docker_v2_CH - DecisionTreeRegressor`)

### Tarefas Concluídas
- [?] Implementação do modelo federado com `DecisionTreeRegressor`.
- [?] Configuração do ambiente Docker (`Dockerfile`, `docker-compose.yml`).
- [?] Implementação da estratégia de agregação (baseada na importância das features).
- [?] Geração de métricas (RMSE, similaridade estrutural) e relatórios (`resumoDTR.md`, artefactos por ronda).
- [?] Implementação da análise de explicabilidade (importância de features, estrutura da árvore).
- [?] Documentação inicial da abordagem (`resumoDTR.md`).

### Tarefas em Andamento
- *Ver tarefas gerais de verificação acima.*

### Tarefas Futuras (Sugestões Pós-Verificação e Transição Dataset)
- [x] (Correção) Uniformizar pré-processamento: Substituída `load_data` em `client_2.py` pela função `load_datasets` (e `CustomDataset`) de `client_1.py`.
- [x] (Verificação Correção) Re-executado `docker-compose up` para DTR: Resultados (RMSE ~66.2k vs ~64.7k) agora muito mais consistentes entre clientes, validando a correção.
- [ ] (Prioridade Média) Avaliar desempenho e explicabilidade no novo dataset.
- [ ] (Prioridade Média/Baixa) Explorar melhorias se necessário/desejado para a dissertação (e.g., Random Forest Federado, otimização de hiperparâmetros).
- [ ] (Prioridade Baixa) Aprofundar análise da evolução da explicabilidade (e.g., métricas quantitativas de convergência).
- [ ] (Prioridade Alta - Dissertação) Documentar resultados e análise na dissertação.

### Notas / Observações
- Implementação parece funcional, mas requer testes.
- Utiliza agregação de importância de features, não agregação direta de modelos.

## Abordagem 2: Linear Regression com Explicabilidade (`flower_docker_v2_CH - ExpCli1Cli2`)

### Tarefas Concluídas
- [?] Implementação do modelo federado com Regressão Linear (PyTorch `nn.Linear`).
- [?] Configuração do ambiente Docker (`Dockerfile`, `docker-compose.yml`).
- [?] Implementação da estratégia de agregação FedAvg.
- [?] Implementação da explicabilidade (SHAP, LIME com instância fixa).
- [?] Geração de métricas (RMSE) e relatórios (`resumo_ExpC1C2.md`, artefactos por ronda).
- [?] Resolução de desafios iniciais (persistência LIME, sincronização, relatório).
- [?] Documentação inicial da abordagem (`resumo_ExpC1C2.md`).

### Tarefas em Andamento
- *Ver tarefas gerais de verificação acima.*

### Tarefas Futuras (Sugestões Pós-Verificação e Transição Dataset)
- [ ] (Prioridade Alta) Investigar o RMSE elevado (235k) após confirmação no novo dataset. Analisar se é limitação do modelo linear, do dataset, ou da implementação.
- [ ] (Prioridade Média) Avaliar desempenho e robustez das explicações (SHAP/LIME) no novo dataset.
- [ ] (Prioridade Média/Baixa) Explorar melhorias no modelo/treino se o RMSE continuar a ser um problema (e.g., taxa de aprendizagem, otimizador, número de épocas locais).
- [ ] (Prioridade Baixa) Explorar outras técnicas de explicabilidade ou estratégias de agregação, se relevante para a dissertação.
- [ ] (Prioridade Alta - Dissertação) Documentar resultados e análise na dissertação.

### Notas / Observações
- RMSE atual muito alto no California Housing, necessita investigação.
- Utiliza FedAvg para agregação.
- Implementa SHAP e LIME para explicabilidade. 