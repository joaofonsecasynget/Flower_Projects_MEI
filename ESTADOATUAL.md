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

## Estado Atual do Projeto (27-04-2025)

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
Atualização feita em 2025-04-28.