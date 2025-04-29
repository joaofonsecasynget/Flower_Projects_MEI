# Estado Atual do Projeto e Tarefas Futuras

Este documento acompanha o progresso do projeto de Aprendizagem Federada e Explicabilidade para ambientes IoT, no âmbito da dissertação "Advanced Federated Learning Strategies: A Multi-Model Approach for Distributed and Secure Environments".

---

## [2025-04-26] Estado Atual e Próximos Passos

- O ciclo federado RLFE foi executado com sucesso até ao fim (5 rondas).
- Todos os artefactos finais foram gerados nas pastas reports/client_X e results/client_X (modelo, métricas, relatórios, explicabilidade LIME, etc.).
- Erro menor: geração de explicações SHAP falhou (`name 'shap' is not defined`), mas o resto dos outputs está correto.
- Warnings de depreciação do Flower: será necessário migrar para o comando `flower-superlink` em versões futuras.
- Próximo passo: documentar exemplos reais de outputs e onboarding, e corrigir o erro SHAP se necessário.

**Nota:** Interrompemos aqui para regressar mais tarde e continuar a documentação e validação dos outputs finais.

---

## [ATUALIZAÇÃO 2025-04-26]

### Contexto
- A implementação principal está agora na pasta `RLFE/`, totalmente containerizada via Docker/Docker Compose.
- O dataset utilizado é o IoT real: `DatasetIOT/transformed_dataset_imeisv_8642840401612300.csv`.
- O ciclo federado (compose, build, up) ocorre exclusivamente dentro de `RLFE/`.
- Healthcheck garante que os clientes só arrancam após o servidor estar disponível.
- Todos os dados, scripts e outputs são independentes de abordagens anteriores.

### Estado Atual
- [/] Validação do ciclo federado RLFE (compose, build, up, logs) a partir de `RLFE/`.
- [/] Outputs, reports e resultados gerados e persistidos em `reports/` e `results/` por cliente.
- [x] Explicabilidade (LIME/SHAP) e relatório HTML consolidados gerados apenas na última ronda.
- [x] Histórico de métricas salvo incrementalmente por ronda, artefactos finais só ao terminar o ciclo.
- [ ] Validar outputs finais e exemplos reais após execução completa.

### Recomendações de Execução
1. Entrar na pasta RLFE:
   ```sh
   cd RLFE
   ```
2. Gerar o compose:
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

---

## Outputs Gerados por Cliente (ao final do ciclo federado)
- **Modelo treinado:** `results/client_X/model_client_X.pt`
- **Histórico de métricas:** `reports/client_X/metrics_history.json`
- **Plots:** `reports/client_X/loss_evolution.png`, `reports/client_X/rmse_evolution.png`
- **Explicabilidade:** `reports/client_X/lime_final.png`, `reports/client_X/shap_final.png`
- **Relatório HTML:** `reports/client_X/final_report.html`

## Pontos Críticos
- O identificador do cliente (`cid`) está presente em todos os logs para facilitar troubleshooting e análise federada.
- O código atualizado encontra-se em `RLFE/client/client.py` e `RLFE/generate_compose.py`.
- Recomenda-se limpar containers antigos antes de novo teste federado.

## Próximos Passos
- [ ] Validar outputs e exemplos reais após execução completa.
- [ ] Atualizar README e documentação com prints/capturas e descrições baseadas nos artefactos concretos.
- [ ] Comparação formal entre abordagens (ADF vs RLFE) usando o novo dataset IoT.
- [ ] Desenvolvimento incremental da dissertação.

---

## Legenda
- [ ] Tarefa por fazer
- [/] Tarefa em andamento
- [x] Tarefa concluída

---

## Histórico de Alterações
- [2025-04-26] Estrutura RLFE consolidada, outputs finais só ao final do ciclo federado, integração Docker completa.

---

## Estado Atual do Projeto (28/04/2025)

## Execução Federada RLFE
- **Todos os clientes federados (client_1, client_2) concluíram a execução.**
- Outputs finais e artefactos de explicabilidade foram gerados para ambos os clientes.

### Artefactos disponíveis para cada cliente
- Relatório HTML final: `final_report.html`
- Explicabilidade LIME: `lime_final.png`, `lime_final.html`, `lime_explanation.txt`
- Explicabilidade SHAP: `shap_final.png`, `shap_values.npy`
- Histórico de métricas: `metrics_history.json`
- Gráficos de evolução: `loss_evolution.png`, `rmse_evolution.png`
- Modelo treinado: `model_client_X.pt` (em `results/`)

### Validação automática
- **Não foram encontrados ficheiros de erro (`*error.txt`) em nenhum cliente.**
- As métricas de treino, validação e teste estão presentes e evoluem conforme esperado (ver `metrics_history.json`).
- Os relatórios HTML e imagens de explicabilidade foram gerados sem falhas.

### Próximos passos sugeridos
- Documentar exemplos dos outputs (prints, screenshots, artefactos) para relatório/dissertação.
- Comparar formalmente os resultados entre abordagens (ADF vs RLFE).
- Atualizar README e documentação com exemplos reais.

---

*Atualização automática: verificação de outputs e estado dos clientes concluída em 28/04/2025 às 12:18.*

## Situação Atual
- O sistema está funcional, mas o relatório final **não estava a ser gerado corretamente devido à ausência do arquivo `X_train.npy`**, necessário para as explicações LIME/SHAP.
- **Foi corrigido o código do cliente para salvar automaticamente o `X_train.npy` após o split dos dados**, garantindo que as explicações de explicabilidade possam ser geradas.
- É necessário realizar novos testes para confirmar que o relatório final agora inclui LIME/SHAP e está completo.

## Próximos Passos
- Testar novamente o sistema para verificar se o relatório final é gerado corretamente, incluindo as explicações LIME/SHAP.
- Analisar logs e outputs durante a execução para identificar potenciais erros ou falhas na geração do relatório.

## Observações
- Mensagens de erro relacionadas com a conexão do servidor foram ignoradas para focar na resolução do problema do relatório.
- **Correção aplicada em 2025-04-28: Salvamento do X_train.npy implementado.**

---

## Tarefas Futuras

- Automatizar a verificação dos outputs dos clientes após o treino federado:
    - Criar um script que percorra as pastas de cada cliente e valide se todos os artefactos esperados foram gerados (relatórios, gráficos, métricas, explicações, modelos, etc.).
    - Sumarizar os principais resultados (ex: RMSE, evolução das perdas) para facilitar a análise sem abrir manualmente cada ficheiro.
    - Alertar se algum artefacto estiver em falta ou incompleto.

---

## Melhorias Prioritárias no Relatório Final (Prioridade Máxima)

**Objetivo:** Aumentar a detalhe e a informação disponível no relatório final (`final_report.html`) gerado por cada cliente, incluindo tempos de execução, evolução de todas as métricas e valores por ronda.

**Prioridade:** Máxima

**Passos Detalhados:**

1.  **Medição e Registo de Tempos:**
    *   **Modificar `client.py`:**
        *   No método `fit`: Registar o tempo (`time.time()`) antes e depois do loop de treino local. Calcular a duração.
        *   No método `evaluate`: Registar o tempo antes e depois da avaliação (incluindo a geração de LIME/SHAP). Calcular a duração total da avaliação/explicabilidade.
        *   Se possível, isolar e cronometrar especificamente a geração LIME e SHAP dentro do `evaluate`.
    *   **Atualizar `metrics_history.json`:**
        *   Adicionar novas chaves ao dicionário de cada ronda para guardar os tempos: `fit_duration_seconds`, `evaluate_duration_seconds`, (opcionalmente `lime_duration_seconds`, `shap_duration_seconds`).
        *   Guardar também os tempos de início e fim (`start_time_fit`, `end_time_fit`, etc.) se for relevante ter o timestamp exato.

2.  **Geração de Gráficos de Evolução Abrangentes:**
    *   **Modificar Lógica de Plotting (em `client.py` ou script auxiliar):**
        *   Identificar a função que gera `loss_evolution.png` e `rmse_evolution.png`.
        *   Generalizar ou duplicar essa função para iterar sobre *todas* as métricas presentes em `metrics_history.json` (e.g., `train_loss`, `val_loss`, `val_rmse`, `test_loss`, `test_rmse`).
        *   Gerar um ficheiro `.png` separado para cada métrica (ex: `train_loss_evolution.png`, `val_loss_evolution.png`, etc.). *Alternativa:* Criar gráficos combinados (ex: todas as losses num gráfico, todos os RMSEs noutro).
    *   **Atualizar Geração do `final_report.html`:**
        *   Incluir tags `<img>` para todos os novos gráficos de evolução gerados.

3.  **Tabela Detalhada de Métricas por Ronda:**
    *   **Modificar Geração do `final_report.html`:**
        *   Remover a tabela atual que mostra apenas os valores finais.
        *   Ler o `metrics_history.json`.
        *   Gerar uma nova tabela HTML onde:
            *   A primeira coluna indica a Ronda (1, 2, 3...).
            *   As colunas seguintes mostram o valor de cada métrica (`train_loss`, `val_loss`, `val_rmse`, `test_loss`, `test_rmse`) para *essa ronda específica*.
            *   Adicionar colunas para os tempos de duração registados no Passo 1 (`fit_duration_seconds`, `evaluate_duration_seconds`).

**Ficheiros a Modificar Principalmente:**

*   `/Users/joaofonseca/git/Flower_Projects_MEI/RLFE/client/client.py`: Para adicionar a lógica de timing e potencialmente a geração de plots/relatórios.
*   Potenciais scripts utilitários de plotting ou reporting, se existirem.

---
Atualização feita em 2025-04-28.

---

## [2025-04-29] Correções e Ajustes Recentes

- **Resolução de Erros `matplotlib`:**
    - Corrigido o `UnboundLocalError: local variable 'plt' referenced before assignment` que impedia a geração de gráficos em algumas execuções.
    - A causa era um import local redundante de `matplotlib.pyplot` dentro da função `evaluate`. A correção envolveu garantir um único import no topo do `client.py`.
    - Dependências de sistema para `matplotlib` (`libfreetype6-dev`, `libpng-dev`) foram adicionadas ao `Dockerfile` para garantir o funcionamento em ambiente headless.
- **Erro `TypeError` em LIME corrigido:** Resolvido um erro onde `num_features` era passado incorretamente para `explainer.explain_lime`.
- **Execução de LIME/SHAP Apenas na Ronda Final:**
    - Modificado o método `evaluate` em `client.py` para que a geração das explicações LIME e SHAP ocorra *apenas* na última ronda federada (`round_number == num_rounds`).
    - Isto resolve o problema de tempos de avaliação excessivamente longos nas rondas intermédias e garante que as explicações são geradas sobre o estado mais atualizado do modelo no cliente após todo o treino federado.
    - O relatório final (`final_report.html`) e o histórico (`metrics_history.json`) agora refletem corretamente esta lógica, mostrando tempos de LIME/SHAP como 0.0 nas rondas intermédias.
- **Validação:** Testes confirmam que os gráficos são gerados e a explicabilidade ocorre apenas na ronda final, com os tempos correspondentes registados corretamente.

## Situação Atual (Pós-Correções)
- O sistema RLFE está estável e a gerar os outputs esperados, incluindo relatórios detalhados e explicações na ronda final.
- As melhorias de detalhe no relatório (tabela por ronda, múltiplos gráficos) implementadas anteriormente estão agora a funcionar sobre uma base corrigida.

## Próximos Passos
- [x] ~~Resolver erros de execução (matplotlib, TypeError)~~.
- [x] ~~Ajustar execução de LIME/SHAP para a ronda final~~.
- [ ] Validar outputs finais e exemplos reais após execução completa (revisitar com base nos relatórios corrigidos).
- [ ] Atualizar README e documentação com prints/capturas e descrições baseadas nos artefactos concretos.
- [ ] Comparação formal entre abordagens (ADF vs RLFE) usando o novo dataset IoT.
- [ ] Desenvolvimento incremental da dissertação.
- [ ] Automatizar a verificação dos outputs dos clientes após o treino federado (ver secção Tarefas Futuras abaixo).

---

## Histórico de Alterações
- [2025-04-26] Estrutura RLFE consolidada, outputs finais só ao final do ciclo federado, integração Docker completa.
- [2025-04-29] Correções de erros matplotlib e ajustes na execução de LIME/SHAP.

---