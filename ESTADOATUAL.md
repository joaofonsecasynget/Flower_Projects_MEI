# Estado Atual do Projeto e Tarefas Futuras

Este documento acompanha o progresso do projeto de Aprendizagem Federada e Explicabilidade para ambientes IoT, no âmbito da dissertação "Advanced Federated Learning Strategies: A Multi-Model Approach for Distributed and Secure Environments".



## Concluído (até 2025-05-01)



### Implementação e Infraestrutura

- [x] Desenvolvimento da estrutura CLFE completamente containerizada via Docker/Docker Compose

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



## Em Andamento



- [ ] Execução de experimentos com diferentes números de clientes para análise comparativa
- [x] Implementação de tabelas otimizadas no relatório HTML para melhor visualização de métricas
- [x] Resolução do problema de métricas faltantes na última ronda
- [ ] Análise aprofundada das features temporais que demonstraram alta importância para o modelo
- [ ] Expansão dos testes para validar robustez em diferentes configurações
- [ ] Migração para comandos mais recentes do Flower (substituição de `start_numpy_client` por `flower-superlink`)
- [ ] Resolução da incompatibilidade de features entre modelo do cliente e dataset original (identificadas 4 features extras)


## Pendente



- [ ] Comparação formal entre CLFE e outras abordagens (ADF) no contexto do dataset IoT
- [ ] Desenvolvimento de visualizações específicas para comparar resultados de diferentes configurações
- [ ] Refinamento da análise de explicabilidade LIME/SHAP no contexto específico do problema IoT
- [ ] Incorporação dos resultados e insights na escrita final da dissertação
- [ ] Avaliação da escalabilidade do sistema com número maior de clientes e rondas



---

## Histórico de Atualizações



### [2025-05-02] Plano para Resolução de Incompatibilidade de Features

#### Problema Identificado
- Identificadas 4 features extras (`extra_feature_0` a `extra_feature_3`) no modelo do cliente que não existem no dataset original
- Essas features recebem pesos significativos no modelo (especialmente `extra_feature_1` com peso -0.053160)
- Features extras aparecem nas explicações LIME/SHAP com importância considerável
- Inconsistência pode comprometer a confiabilidade das explicações

#### Plano de Resolução da Incompatibilidade de Features e Explicabilidade

#### Fase de Investigação 
1. Determinar a natureza exata do problema (features diferentes entre o dataset original e o modelo do cliente)
2. Identificar a fonte da incompatibilidade (transformações temporais aplicadas em client.py)
3. Verificar o impacto em: (a) explicabilidade, (b) correlação com o resultado, (c) visualização

#### Fase de Análise de Impacto 
1. Quantificar os desvios entre modelos de interpretabilidade e o modelo real
2. Avaliar o impacto nas explicações LIME/SHAP
3. Determinar quais algoritmos são mais sensíveis à incompatibilidade de features

#### Fase de Correção 
1. Implementar correções no data_loader.py para usar nomes corretos das features
2. Garantir que os valores temporais sejam corretamente calculados e usados
3. Alinhar a dimensionalidade e nomes das features entre dataset e modelo

#### Fase de Melhoria de Explicabilidade (Concluída) ✓
1. ✓ Criar sistema centralizado de metadados de features (feature_metadata.py)
2. ✓ Implementar script de geração automática de metadados (generate_feature_metadata.py)
3. ✓ Integrar sistema de metadados com data_loader.py
4. ✓ Atualizar html_report.py para usar metadados e exibir valores originais
5. ✓ Aprimorar visualizações LIME/SHAP para apresentar corretamente valores reais e não normalizados

#### Fase de Validação e Documentação (Em Andamento)
1. ✓ Documentar o sistema de metadados (README_METADADOS.md)
2. Criar testes para garantir consistência de features
3. Atualizar documentação do projeto
4. ✓ Validar explicações antes e depois das mudanças
5. Documentar processo de explicabilidade end-to-end com rastreabilidade de features

### [2025-05-02] Análise do Modelo e Treino Federado

#### Descobertas sobre o Modelo Atual
- O modelo atual utiliza regressão linear (LinearRegressionModel) sem função de ativação específica para classificação
- A perda utilizada é MSE (Mean Squared Error), típica de problemas de regressão, não de classificação binária
- Os dados de entrada (features) são normalizados usando StandardScaler, mas não há evidência de normalização do target
- Os valores de previsão são significativamente altos (na ordem de -900.000), incompatíveis com um classificador binário

#### Implicações para Explicabilidade e Interpretabilidade
- Os valores extremamente altos das previsões dificultam a interpretação das contribuições das features
- O limiar arbitrário de classificação (prediction < 0.5) não é apropriado para o modelo
- As explicações LIME/SHAP podem estar sendo afetadas pelo modelo de regressão vs classificação

#### Próximos Passos para o Treino Federado
1. Revisar a arquitetura do modelo considerando:
   - Transformação para classificador binário (adicionando sigmoid ou limiar ajustado)
   - Normalização adequada dos valores target durante o treinamento
   - Ajuste do critério de perda para Binary Cross Entropy (BCE)

2. Implementar melhorias no processo de treino federado:
   - Verificar estratégia de agregação no servidor para modelo de regressão vs classificação
   - Ajustar hiperparâmetros para otimizar convergência em configuração federada
   - Implementar métricas específicas de classificação (accuracy, precision, recall, F1)

3. Atualizar pipeline de explicabilidade:
   - Ajustar interpretação das previsões para considerar se o modelo é de regressão ou classificação
   - Escalar corretamente os valores das contribuições das features para melhor interpretabilidade
   - Documentar claramente na interface de explicabilidade que tipo de modelo está sendo usado

### [2025-05-02] Conversão de Regressão para Classificação (RLFE → CLFE)

#### Mudanças Implementadas
- [x] Alterado o nome do projeto de RLFE (Regressão Linear Federada Explicável) para CLFE (Classificação Linear Federada Explicável)
- [x] Modelo renomeado de `LinearRegressionModel` para `LinearClassificationModel`
- [x] Adicionada função de ativação sigmoid na camada de saída do modelo: `return torch.sigmoid(self.linear(x))`
- [x] Substituída a função de perda MSE por BCE (Binary Cross Entropy): `criterion = nn.BCELoss()`
- [x] Implementadas novas métricas de classificação no cliente:
  - Accuracy: exatidão geral da classificação
  - Precision: precisão para a classe positiva (ataques)
  - Recall: sensibilidade na detecção de ataques
  - F1-score: média harmônica entre precisão e recall
- [x] Atualizada a estratégia de agregação no servidor para processar as novas métricas
- [x] Mantidas métricas RMSE para compatibilidade com visualizações existentes
- [x] Atualizados README.md e ESTADOATUAL.md para refletir a nova abordagem

#### Benefícios Esperados
- Previsões em formato de probabilidade [0,1], mais interpretáveis 
- Limiar de classificação de 0.5 agora apropriado para o modelo sigmoidal
- Métricas mais relevantes para o problema de classificação binária
- Explicações LIME/SHAP mais precisas e interpretáveis
- Alinhamento correto entre o tipo de problema (classificação binária) e a implementação do modelo

#### Próximos Passos
- Testar o modelo convertido com diferentes números de clientes
- Avaliar o impacto da mudança nas explicações LIME/SHAP
- Comparar o desempenho do classificador linear com outras abordagens
- Documentar os resultados na dissertação

### [2025-05-02] Melhorias na Interface do Relatório HTML e Correções nas Métricas

#### Problemas Identificados
- Tabela de métricas excessivamente larga com muitas colunas, dificultando a leitura
- Ausência de métricas de treino (train_loss e fit_duration) na última ronda
- Problemas de posicionamento (ordem) das colunas na tabela de métricas

#### Soluções Implementadas
1. **Divisão da tabela de métricas em três tabelas temáticas:**
   - Tabela 1: Métricas de Treino e Validação (train_loss, val_loss, val_rmse, val_accuracy, etc.)
   - Tabela 2: Métricas de Teste (test_loss, test_rmse, test_accuracy, test_precision, etc.)
   - Tabela 3: Tempos de Processamento (fit, evaluate, lime, shap)
   
2. **Correção das métricas na última ronda:**
   - Implementação de verificação no método `evaluate()` para detectar a última ronda
   - Execução de treinamento adicional quando necessário para garantir o registro completo de métricas
   - Registro de logs específicos para essa operação de backup

3. **Organização otimizada das colunas:**
   - Agrupamento de métricas por tipo (treino, validação, teste, tempos)
   - Posicionamento de métricas à esquerda e tempos à direita
   - Simplificação dos nomes de colunas para melhor legibilidade

#### Resultados das Execuções
- Testado com 2 clientes de 20 partições totais: desempenho perfeito (F1-score ~1.0)
- Testado com 3 clientes de 6 partições totais: desempenho muito alto (F1-score ~0.98)
- Convergência extremamente rápida em todos os testes
- Particionamento estratificado garantindo representatividade dos targets em cada cliente

#### Próximos Passos
- Atualização da documentação com os resultados obtidos
- Desenvolvimento do módulo final de comparação formal
- Testes com números maiores de clientes para analisar o impacto na federação
- Análise das explicabilidades LIME/SHAP para compreender as razões do excelente desempenho