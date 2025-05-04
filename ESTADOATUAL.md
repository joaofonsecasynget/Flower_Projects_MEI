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

### [2025-05-04] Refinamentos na Interface de Relatório e Correções de Bugs

#### Problemas Identificados
- Gráficos de métricas mostravam 10 rondas quando na realidade eram apenas 5
- Duplicação de valores no histórico de métricas de classificação (accuracy, precision, recall, f1)
- Organização subótima dos gráficos no relatório HTML final
- Duplicação do gráfico "Comparação de Accuracy" na seção de evolução

#### Melhorias Implementadas
1. **Correção de duplicação de métricas:**
   - Corrigida a lógica que armazenava as métricas de classificação no histórico
   - Implementada verificação para evitar duplicação de registros por ronda
   - Melhorada a consistência entre número de valores em diferentes métricas

2. **Reorganização dos gráficos no relatório HTML:**
   - Tempos de explicabilidade (LIME/SHAP) movidos para o início da seção de explicabilidade
   - Implementado layout de duas colunas para visualização lado a lado de LIME e SHAP
   - Otimizada a apresentação para melhor aproveitamento do espaço e visualização

3. **Refatoração do código:**
   - Simplificação da geração de plot_files para evitar filtros desnecessários
   - Melhoria na organização e readabilidade do código
   - Documentação das decisões de design com comentários

#### Benefícios 
- Representação mais fiel do número real de rondas nos gráficos
- Melhor experiência de visualização com layout de colunas para LIME/SHAP
- Organização mais lógica dos gráficos no relatório HTML
- Redução da duplicação e simplificação do código

#### Status Atual
- Todas as melhorias estão implementadas e testadas
- Relatório HTML agora apresenta corretamente os dados e gráficos
- Experiência de visualização significativamente melhorada

#### Próximos Passos
- Continuar a execução de experimentos com diferentes números de clientes
- Incorporar os insights obtidos dos relatórios aprimorados na dissertação
- Avaliar se são necessárias melhorias adicionais na interface de relatório

### [2025-05-04] Plano de Melhorias para Gráficos de Explicabilidade

#### Problemas Identificados com Explicabilidade LIME/SHAP

1. **LIME - Identificação da Instância:**
   - A instância específica sendo explicada pelo LIME não está identificada no relatório
   - Não há informações sobre o índice, características ou classificação real/prevista da instância
   - Impossível contextualizar a explicação sem conhecer a instância analisada

2. **LIME - Truncamento de Nomes de Features:**
   - Nomes das features estão truncados no gráfico `lime_final.png`
   - Dificulta a identificação precisa de quais features estão influenciando a previsão
   - Compromete a interpretabilidade do gráfico

3. **LIME - Apenas Valores Negativos:**
   - O gráfico LIME atual mostra apenas valores de contribuição negativos
   - Pode indicar um problema na forma como a explicação está sendo gerada ou visualizada
   - Normalmente esperaríamos ver contribuições positivas (que aumentam a probabilidade da classe prevista) e negativas (que diminuem)

4. **SHAP - Inconsistência na Apresentação de Features:**
   - Foi indicado que SHAP mostraria a distribuição completa de todas as features, mas o gráfico não mostra todas
   - Possível conflito entre a documentação/explicação e a implementação real
   - Necessário esclarecer se é uma limitação intencional ou um problema

#### Análise Técnica e Causas Prováveis

1. **Problema LIME - Identificação:**
   - Provável causa: Na função `generate_explainability()`, a instância é selecionada arbitrariamente (índice 0 por padrão)
   - Não há código para extrair e exibir metadados sobre a instância no relatório HTML

2. **Problema LIME - Nomes Truncados:**
   - Provável causa: Configuração padrão da biblioteca LIME para visualização
   - Falta de personalização no tamanho do gráfico ou nas margens para acomodar nomes longos

3. **Problema LIME - Apenas Valores Negativos:**
   - Possíveis causas:
     * Configuração incorreta da classe positiva no LIME (invertendo a interpretação)
     * A instância selecionada tem características que reduzem a probabilidade da classe prevista
     * Bug na geração da visualização LIME

4. **Problema SHAP - Features Limitadas:**
   - Possíveis causas:
     * Configuração padrão do `summary_plot` do SHAP que limita o número de features mostradas
     * Filtragem intencional não documentada
     * Diferença entre a quantidade total de features e as que têm impacto significativo

#### Plano de Ação Detalhado

1. **Melhorias para LIME:**
   
   a) **Identificação da Instância:**
   ```python
   # Modificar generate_explainability para incluir:
   # - Permitir seleção aleatória com seed ou índice específico
   # - Extrair características básicas da instância
   # - Salvar informações sobre a instância em um arquivo adicional
   # - Incluir metadados no relatório HTML
   ```
   
   b) **Correção do Truncamento de Nomes:**
   ```python
   # Ajustar a função de geração do gráfico LIME:
   # - Aumentar o tamanho da figura para acomodar nomes longos
   # - Ajustar margens/padding do gráfico
   # - Implementar tratamento de nomes longos (abreviação inteligente ou quebra de linha)
   ```
   
   c) **Investigação dos Valores Negativos:**
   ```python
   # 1. Verificar a configuração da classe positiva no explainer LIME
   # 2. Analisar diferentes instâncias para verificar se o problema persiste
   # 3. Ajustar parâmetros de visualização se necessário
   # 4. Documentar o comportamento esperado com base na interpretação correta
   ```

2. **Melhorias para SHAP:**
   
   a) **Clarificação da Visualização de Features:**
   ```python
   # 1. Verificar a documentação do SHAP sobre limites padrão
   # 2. Implementar controle explícito do número de features mostradas
   # 3. Documentar o comportamento real vs. esperado
   # 4. Adicionar opção para visualizar todas as features ou apenas as mais importantes
   ```
   
   b) **Adição de Visualizações Complementares:**
   ```python
   # Implementar visualizações adicionais que mostram outras perspectivas dos valores SHAP:
   # - Gráfico de barras com todas as features ordenadas por importância
   # - Opção para filtrar por grupos de features
   ```

3. **Melhorias na Documentação e Relatório:**
   
   a) **Esclarecimentos no Relatório HTML:**
   ```html
   <!-- Adicionar seções explicativas: -->
   <div class="explanation-notes">
     <h4>Notas sobre LIME:</h4>
     <p>Esta explicação refere-se à instância [ID]. Valores negativos indicam características que reduzem a probabilidade de classificação como ataque.</p>
     
     <h4>Notas sobre SHAP:</h4>
     <p>Este gráfico mostra as [X] features mais importantes. Cores representam o valor da feature (vermelho = alto, azul = baixo).</p>
   </div>
   ```
   
   b) **Documentação do Código:**
   ```python
   # Melhorar documentação em-código para:
   # - Esclarecer a interpretação correta dos valores LIME/SHAP
   # - Documentar parâmetros e comportamentos padrão
   # - Explicar limitações e configurações personalizáveis
   ```

#### Priorização e Cronograma

1. **Alta Prioridade (Implementação Imediata):**
   - Identificação da instância LIME no relatório
   - Correção do problema de truncamento de nomes de features
   - Investigação e correção dos valores apenas negativos no LIME

2. **Média Prioridade:**
   - Clarificação e documentação da visualização SHAP
   - Melhorias na documentação do relatório HTML

3. **Baixa Prioridade (Melhorias Futuras):**
   - Visualizações SHAP adicionais
   - Interface para seleção interativa de instâncias para explicação

Este plano será executado como parte das melhorias contínuas do projeto CLFE, com objetivo de aumentar a interpretabilidade e utilidade das explicações LIME/SHAP, fundamentais para a transparência do modelo de detecção de ataques IoT.

### [2025-05-04] Problemas Identificados com Explicabilidade LIME/SHAP

#### Problemas Identificados com Explicabilidade LIME/SHAP

- **LIME - Truncamento dos Nomes das Features**: Os nomes das features apareciam truncados na visualização LIME, dificultando a identificação precisa das variáveis importantes.
- **LIME - Identificação da Instância**: Não havia informação clara sobre qual instância estava sendo analisada e seu índice original no dataset.
- **LIME - Valores das Features**: Os valores apresentados eram normalizados, dificultando a interpretação no contexto do domínio do problema.
- **LIME - Somente Valores Negativos**: A visualização LIME mostrava apenas contribuições negativas, com esquema de cores inadequado.
- **SHAP - Inconsistências na Apresentação das Features**: Algumas features podem não estar sendo exibidas corretamente na visualização SHAP.

#### Melhorias Implementadas na Explicabilidade (2025-05-04)

1. **Correção do Truncamento de Nomes das Features no LIME**:
   - Implementada visualização LIME personalizada com maior tamanho (12x8) para acomodar nomes completos
   - Ajustado o layout e formatação para melhor visualização
   - Adicionada legenda explicativa sobre o significado das cores (verde para normal, vermelho para ataque)

2. **Melhoria na Identificação da Instância**:
   - Implementada seleção aleatória de instâncias para análise
   - Adicionado rastreamento e exibição do índice original no dataset
   - Apresentação clara separando "Índice na amostra" do "Índice no dataset original"

3. **Correção dos Valores das Features**:
   - Implementado carregamento do dataset original para obter valores não normalizados
   - Exibição dos valores originais (não normalizados) no relatório
   - Adicionada indicação clara de que os valores apresentados são os originais

4. **Apresentação Visual Aprimorada**:
   - Implementado esquema de cores correto (verde=normal, vermelho=ataque) 
   - Layout de duas colunas para LIME e SHAP para melhor aproveitamento do espaço
   - Seção dedicada com informações detalhadas sobre a instância analisada

#### Próximos Passos para Explicabilidade

- Investigar e corrigir o problema de valores apenas negativos no LIME
- Verificar e corrigir as inconsistências na apresentação de features no SHAP
- Avaliar a possibilidade de integrar mais aspectos do sistema especializado de explicabilidade (`CLFE/explainability/`) no fluxo principal