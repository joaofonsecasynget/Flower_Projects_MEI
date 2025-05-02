# Estado Atual do Projeto e Tarefas Futuras

Este documento acompanha o progresso do projeto de Aprendizagem Federada e Explicabilidade para ambientes IoT, no âmbito da dissertação "Advanced Federated Learning Strategies: A Multi-Model Approach for Distributed and Secure Environments".



## ✅ Concluído (até 2025-05-01)



### Implementação e Infraestrutura

- [x] Desenvolvimento da estrutura RLFE completamente containerizada via Docker/Docker Compose

- [x] Configuração do ciclo federado com servidor central e múltiplos clientes

- [x] Implementação de healthchecks para garantir inicialização ordenada dos componentes

- [x] Mecanismo de geração automática de docker-compose através do script `generate_compose.py`

- [x] Integração do dataset IoT real: `DatasetIOT/transformed_dataset_imeisv_8642840401612300.csv`



### Processamento de Dados e Modelagem

- [x] Implementação do particionamento estratificado para distribuição equilibrada do target entre clientes

- [x] Extração de features temporais do campo `_time` (hora, dia da semana, dia, mês, fim de semana)

- [x] Tratamento robusto de diferentes formatos de timestamp usando ISO8601

- [x] Modelo de detecção de ataques com métricas de avaliação completas



### Explicabilidade e Visualização

- [x] Implementação de LIME para explicações locais do modelo

- [x] Implementação de SHAP para explicações globais do modelo

- [x] Geração de visualizações por categoria específica de feature (dl_bitrate, ul_bitrate, etc.)

- [x] Layout aprimorado para visualizações de explicabilidade no relatório HTML

- [x] Gráficos de importância temporal e de features derivadas de timestamp

- [x] Script para explicabilidade interativa de instâncias individuais (`explain_instance.py`)



### Artefatos e Documentação

- [x] Geração de relatórios HTML consolidados por cliente

- [x] Monitoramento e registro de métricas por ronda

- [x] Salvamento de artefatos em diretórios organizados (modelo, métricas, explicabilidade)

- [x] Documentação README.md completa e detalhada

- [x] Histórico de atualizações no ESTADOATUAL.md



## 🔄 Em Andamento



- [ ] Execução de experimentos com diferentes números de clientes para análise comparativa

- [ ] Análise aprofundada das features temporais que demonstraram alta importância para o modelo

- [ ] Expansão dos testes para validar robustez em diferentes configurações

- [ ] Migração para comandos mais recentes do Flower (substituição de `start_numpy_client` por `flower-superlink`)



## 📋 Pendente



- [ ] Comparação formal entre RLFE e outras abordagens (ADF) no contexto do dataset IoT

- [ ] Desenvolvimento de visualizações específicas para comparar resultados de diferentes configurações

- [ ] Refinamento da análise de explicabilidade LIME/SHAP no contexto específico do problema IoT

- [ ] Incorporação dos resultados e insights na escrita final da dissertação

- [ ] Avaliação da escalabilidade do sistema com número maior de clientes e rondas



---

## Histórico de Atualizações



### [2025-05-02] Implementação de Explicabilidade Interativa

- [x] Desenvolvido script `explain_instance.py` para análise detalhada de instâncias individuais
- [x] Implementada capacidade de selecionar qualquer instância (por índice ou aleatoriamente)
- [x] Geração de explicações LIME e SHAP para instâncias específicas
- [x] Visualizações customizadas para entender o comportamento do modelo em nível individual
- [x] Arquitetura flexível para carregar modelos e dados de diferentes clientes federados

### [2025-05-01] Melhorias de Explicabilidade e Correção de Bugs

#### Correções de Bugs

- [x] Corrigido o problema de extração de features temporais do campo `_time`, assegurando que essas features sejam extraídas antes da remoção da coluna original

- [x] Implementada manipulação mais robusta de formatos de timestamp utilizando ISO8601

- [x] Resolvido erro na geração do relatório HTML final devido a problemas com sintaxe CSS nas f-strings



#### Melhorias na Visualização da Explicabilidade

- [x] Redesenhado o layout das visualizações de explicabilidade no relatório HTML

- [x] Implementada estrutura CSS grid responsiva para exibição mais lógica dos gráficos

- [x] Confirmado que as categorias específicas de features são corretamente utilizadas para agregação



#### Descobertas Relevantes

- [x] Features temporais (categorizadas como "other") têm significativa importância no modelo

- [x] Visualizações específicas de explicabilidade temporal agora aparecem corretamente nos relatórios

- [x] Particionamento estratificado mantém a proporção global de registros normais vs. ataques



### [2025-04-26] Validação do Ciclo Federado



- [x] Ciclo federado RLFE executado com sucesso (5 rondas)

- [x] Artefatos finais gerados nas pastas reports/client_X e results/client_X

- [x] Adoção da estrutura containerizada via Docker/Docker Compose

- [x] Integração do dataset IoT real



_Nota: As seções anteriores foram mantidas como histórico, mas todo o trabalho mencionado nelas já foi concluído e está refletido na seção "Concluído" acima._