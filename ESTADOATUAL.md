# Estado Atual do Projeto e Tarefas Futuras

Este documento acompanha o progresso do projeto de Aprendizagem Federada e Explicabilidade para Previsão de Preços Imobiliários, no âmbito de uma dissertação de mestrado.

## Legenda

- [ ] Tarefa por fazer
- [/] Tarefa em andamento
- [x] Tarefa concluída
- [?] Tarefa concluída, mas a necessitar de verificação/teste

## Tarefas Gerais / Próximos Passos Principais

- [x] **Verificação das Implementações Atuais:**
    - [x] 1.1: Testar execução `docker-compose up` para Árvore de Decisão Federada (DecisionTreeRegressor, scikit-learn).
    - [x] 1.2: Verificar outputs (logs, métricas, relatórios) da Árvore de Decisão Federada (DecisionTreeRegressor, scikit-learn) vs `resumoDTR.md`. (Logs, métricas e gráficos de evolução OK. Gráfico de Soma Importância mantido, mas pouco útil; clarificado cálculo/apresentação da Similaridade Estrutural — agora aparece no relatório como valor final e evolução no gráfico.)
    - [x] 1.3: Testar execução `docker-compose up` para Regressão Linear Federada com Explicabilidade via LIME e SHAP (PyTorch).
    - [x] 1.4: Verificar outputs (logs, métricas, relatórios) da Regressão Linear Federada com Explicabilidade via LIME e SHAP (PyTorch) vs `resumo_ExpC1C2.md`. (Logs, métricas (RMSE, tempo), gráficos LIME e SHAP gerados corretamente. Confirmado RMSE elevado e convergência lenta. Relatórios de explicabilidade completos e comparáveis.)
    - [x] 1.5: Analisar consistência do pré-processamento (`utils.py` em `client_1`/`client_2` de ambas as abordagens). (Ambas as abordagens usam agora métodos consistentes de carregamento, tratamento de NaN, encoding e particionamento.)
    - [x] 1.6: Verificar distribuição dos dados entre clientes. **IMPORTANTE**: Cada cliente usa uma partição distinta do dataset (índices diferentes e partições fixas/reprodutíveis via seed), simulando aprendizagem federada realista.
- [ ] **Comparação Formal das Abordagens:**
    - [ ] Gerar tabela comparativa de métricas e explicabilidade entre Árvore de Decisão Federada (DecisionTreeRegressor, scikit-learn) e Regressão Linear Federada com Explicabilidade via LIME e SHAP (PyTorch).
    - [ ] Documentar limitações observadas (ex: RMSE elevado na regressão linear, explicabilidade limitada pelo modelo).
    - [ ] Definir critérios claros para a comparação (desempenho, estabilidade, robustez da explicabilidade, tempo de execução, etc.).
    - [ ] Desenvolver um protocolo de teste padronizado para aplicar a ambas as abordagens.
    - [ ] Executar as experiências comparativas.
- [ ] **Transição para Novo Dataset:**
    - [ ] Definir/Obter o novo dataset alvo para a dissertação.
    - [ ] Adaptar o código de carregamento e pré-processamento para o novo dataset.
- [ ] **Desenvolvimento da Dissertação:**
    - [ ] Estruturar o documento da dissertação.
    - [/] Escrever secções relevantes à medida que o trabalho progride (e.g., Introdução, Revisão da Literatura, Metodologia baseada nas implementações atuais).

## Estado Atual do Projeto DTR (Árvore de Decisão Federada (DecisionTreeRegressor, scikit-learn))

### Objetivos Atuais
- [x] Implementar Árvore de Decisão Federada (DecisionTreeRegressor, scikit-learn) com Flower.
- [x] Adicionar funcionalidades de explicabilidade (visualização de árvores, importância de features).
- [x] Gerar relatórios individuais por cliente (HTML).
- [x] Uniformizar pré-processamento entre clientes para garantir consistência.
- [ ] Investigar e resolver inconsistência na geração do gráfico `explainability_evolution.png` no Cliente 2.

### Últimas Alterações e Verificações
- **Uniformização do Pré-processamento:**
    - Substituída a função `load_data` em `client_2.py` pela implementação de `client_1.py` (`load_datasets` e `CustomDataset`).
    - Re-execução confirmou resultados de RMSE mais consistentes entre os clientes (Client 1: ~66.2k, Client 2: ~64.7k), validando a correção.
- **Depuração da Geração de Gráficos:**
    - Identificado e corrigido um erro lógico (`prev_tree = curr_tree` em falta) no cálculo da similaridade em `client_2/utils.py`.
    - Identificado um problema na geração do gráfico `explainability_evolution.png` no Cliente 2, que parecia falhar silenciosamente durante a gravação/fecho da figura Matplotlib.
    - Forçado o uso do backend 'Agg' do Matplotlib em ambos os clientes. Os logs subsequentes indicaram sucesso na geração do gráfico no Cliente 2, mas o ficheiro continua a não ser criado.
- **Particionamento dos Dados:**
    - [x] Corrigido o particionamento dos dados em client_2.py para garantir partições fixas e distintas entre clientes, mantendo a seed fixa (42) e sem obrigatoriedade de usar todos os dados.

### Problemas Conhecidos
- **Geração Inconsistente do Gráfico de Evolução (Cliente 2):** Apesar da depuração e da aplicação do backend 'Agg', o ficheiro `explainability_evolution.png` não é consistentemente gerado na pasta `reports` do Cliente 2, embora os logs indiquem sucesso. A causa raiz permanece incerta e requer investigação adicional (possivelmente relacionada com o ambiente Docker específico ou interações subtis com Matplotlib).

### Tarefas Futuras (Sugestões Pós-Verificação e Transição Dataset)
    - [ ] (Prioridade Média) Avaliar desempenho e explicabilidade no novo dataset.
    - [ ] (Prioridade Média/Baixa) Explorar melhorias se necessário/desejado para a dissertação (e.g., Random Forest Federado, otimização de hiperparâmetros).
    - [ ] (Prioridade Baixa/Investigação) Resolver em definitivo o problema da geração do gráfico no Cliente 2.

### Próximos Passos Imediatos (Pausa)
- Pausar o desenvolvimento para reavaliação e planeamento.

## Abordagem 2: Regressão Linear Federada com Explicabilidade via LIME e SHAP (PyTorch)

### Tarefas Concluídas
- [?] Implementação do modelo federado com Regressão Linear Federada com Explicabilidade via LIME e SHAP (PyTorch).
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

## Estrutura Modular RLFE
- Nova pasta `RFLE/` criada para a abordagem Regressão Linear Federada Explicável (RLFE).
- Cada cliente é instanciado a partir de um único script parametrizado (`client.py`), recebendo `cid` e `num_clients` como argumentos.
- O particionamento do dataset é feito "on-the-fly" por cada cliente, garantindo flexibilidade e reprodutibilidade.
- Pastas de output dedicadas para cada cliente em `reports/client_N` e `results/client_N`.
- O arranque de múltiplos clientes será feito automaticamente via Docker Compose, usando a variável `NUM_CLIENTS`.
- Volumes Docker serão usados para persistência dos resultados e partilha do dataset.
- Integração do pipeline federado e de explicabilidade prevista após validação da estrutura base.

## Abreviaturas das Abordagens
- **ADF**: Árvore de Decisão Federada (DecisionTreeRegressor, scikit-learn)
- **RLFE**: Regressão Linear Federada Explicável (PyTorch, LIME/SHAP)

## Próximos Passos Recomendados (Novo Dataset)

- Criar um subconjunto reduzido do novo dataset (`transformed_dataset_imeisv_8609960480666910.csv`) com 2 000 a 5 000 linhas (amostragem aleatória ou estratificada), para facilitar o desenvolvimento e teste rápido das abordagens.
- Desenvolver, testar e ajustar as abordagens federadas (pré-processamento, treino, avaliação) usando este subconjunto.
- Após estabilização e validação do código, escalar para o dataset completo (74 411 linhas) para análise final de desempenho e generalização.
- (Opcional) Automatizar a criação do subconjunto para garantir reprodutibilidade e facilitar futuras iterações.

Estes passos permitem acelerar o desenvolvimento, garantir testes eficientes e só utilizar o dataset completo quando o código estiver maduro.

## Preparação do Novo Dataset

- O dataset de testes iniciais (`ds_testes_iniciais.csv`) encontra-se preparado para utilização, desde que se realize:
    - Substituição de valores -1 por NaN;
    - Remoção de colunas não relevantes (ex: indice, _time, imeisv);
    - Confirmação do target (`attack`);
    - Normalização/padronização das variáveis numéricas.
- Recomenda-se preencher os valores em falta na tabela comparativa após os primeiros testes com o novo dataset.
- Manter o desenvolvimento e debugging sobre o subconjunto reduzido antes de escalar para o dataset completo.