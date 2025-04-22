# Treino Federado com Árvores de Decisão para Previsão de Preços de Imóveis

## 📋 Sumário Executivo

Este projeto implementa um sistema de treino federado utilizando o framework Flower para prever preços de imóveis na Califórnia. O sistema utiliza dois clientes que treinam modelos de árvores de decisão de forma distribuída, permitindo aprendizagem colaborativa sem compartilhamento direto dos dados.

## 🎯 Objetivos

1. Implementar um sistema de treino federado funcional
2. Manter a privacidade dos dados entre os clientes
3. Gerar previsões precisas de preços de imóveis
4. Fornecer interpretabilidade através de visualizações e métricas claras

## 🛠 Estratégia Adotada

### Arquitetura do Sistema
- **Servidor Central**: Coordena o processo de treino federado
- **2 Clientes**: Treinam modelos localmente com seus próprios dados
- **Comunicação**: Via gRPC usando o framework Flower
- **Modelo**: Árvore de Decisão (DecisionTreeRegressor do scikit-learn)

### Distribuição e Fluxo de Dados
1. **Dataset Base**:
   - California Housing Dataset: dados demográficos e habitacionais de regiões da Califórnia
   - Ambos os clientes utilizam o mesmo conjunto de dados base, mas com processamentos independentes

2. **Particionamento**:
   - **Cliente 1**: 
     - Implementa a função `load_datasets` que cria partições específicas usando a semente 42
     - Divide os dados em partições para cada cliente de forma determinística
   - **Cliente 2**: 
     - Implementa sua própria função `load_data` de carregamento 
     - Usa a mesma semente aleatória 42 para manter consistência

3. **Pré-processamento Local**:
   - Todos os dados são normalizados usando `StandardScaler`
   - Valores ausentes são preenchidos com a média da coluna
   - A coluna categórica `ocean_proximity` é transformada em valores numéricos
   - Divisões: 80% treino, 10% validação, 20% teste

4. **Consistência entre Rondas**:
   - Os dados utilizados em cada ronda são exatamente os mesmos
   - Isto garante que a evolução da explicabilidade reflete apenas o aprendizado do modelo
   - As mudanças observadas não são influenciadas por variações nos dados

### Características do Modelo
- Profundidade máxima: 10
- Features utilizadas:
  - longitude
  - latitude
  - idade média da habitação
  - total de divisões
  - total de quartos
  - população
  - agregados familiares
  - rendimento médio
  - proximidade ao oceano

## 📊 Resultados Obtidos

### Métricas de Desempenho
- **Cliente 1**:
  - Training Loss: 1.84B
  - Validation Loss: 3.98B
  - Evaluation Loss: 4.39B
  - RMSE: 66,234.26
  - Similaridade entre árvores: 0.95 (alta consistência estrutural)

- **Cliente 2**:
  - Training Loss: 1.84B
  - Validation Loss: 4.12B
  - Evaluation Loss: 3.77B
  - RMSE: 61,409.84
  - Similaridade entre árvores: 0.97 (alta consistência estrutural)

### Tempo de Execução
- Tempo total: 1.16 segundos
- Tempo médio por ronda: ~0.23 segundos

### Evolução da Explicabilidade
O sistema demonstrou uma evolução consistente da explicabilidade ao longo das 5 rondas:

1. **Similaridade Estrutural**:
   - Cliente 1: 0.95 (alta consistência)
   - Cliente 2: 0.97 (alta consistência)
   - Indica que as árvores mantêm uma estrutura estável entre rondas

2. **Convergência do Modelo**:
   - As métricas de perda mostram estabilidade
   - RMSE consistentes entre rondas
   - Alta similaridade estrutural indica convergência

3. **Eficácia da Federação**:
   - Ambos os clientes alcançaram desempenho similar
   - RMSE entre 61k e 66k dólares
   - Consistência nas previsões entre clientes

#### Artefactos Gerados por Ronda
1. **Estrutura da Árvore**: 
   - Ficheiros de texto contendo a representação completa da árvore
   - Documentam todas as regras de decisão, valores de divisão e predições
   - Formato: `tree_structure_roundX.txt`

2. **Importância das Características**:
   - Gráficos mostrando a relevância relativa de cada característica
   - Permitem identificar mudanças na importância ao longo das rondas
   - Formato: `feature_importance_roundX.png`

3. **Métricas de Desempenho**:
   - Valores detalhados para cada fase (treino, validação, teste)
   - Armazenados em JSON para análises comparativas
   - Incluem MSE e RMSE para cada ronda

4. **Gráfico de Evolução da Explicabilidade**:
   - Visualização que combina múltiplas métricas em um único gráfico
   - Painel superior: evolução da soma da importância das características
   - Painel intermédio: evolução da complexidade da árvore
   - Painel inferior: nova métrica de similaridade estrutural e RMSE
   - Permite correlacionar desempenho com explicabilidade
   - Formato: `explainability_evolution.png`

5. **Métrica de Similaridade Estrutural**:
   - Quantificação objetiva da similaridade entre árvores de rondas consecutivas
   - Baseada na concordância das previsões em pontos aleatórios do espaço de características
   - Valores entre 0 (completamente diferentes) e 1 (idênticas)
   - Permite identificar quando o modelo estabiliza ou sofre alterações significativas

6. **Relatório Integrado Final**:
   - HTML agregando todos os artefactos anteriores
   - Facilita comparações visuais entre rondas
   - Formato: `final_report.html`

#### Análise da Evolução
O sistema permite analisar vários aspetos da evolução do modelo:

1. **Convergência Estrutural**:
   - Observação da estabilização da estrutura da árvore
   - Análise da consistência nos nós de decisão principais
   - Monitorização da métrica de similaridade para verificar a estabilidade entre rondas
   - Identificação precisa do momento em que o modelo converge estruturalmente

2. **Dinâmica de Importância**:
   - Identificação de características que ganham ou perdem relevância
   - Monitorização da estabilidade da hierarquia de importância

3. **Evolução das Regras de Decisão**:
   - Análise de como as regras se tornam mais específicas ou generalistas
   - Observação da convergência dos valores de corte em nós similares
   - Correlação entre mudanças nas regras e na similaridade estrutural

4. **Correlação com Desempenho**:
   - Relação entre mudanças estruturais e melhorias de métricas
   - Identificação de padrões indicativos de maior generalização
   - Análise do impacto da estabilidade estrutural (alta similaridade) no desempenho preditivo

5. **Visualização Integrada**:
   - O novo gráfico de evolução da explicabilidade permite uma visão holística
   - Facilita a identificação de correlações entre complexidade, importância, similaridade e desempenho
   - Ajuda a responder questões como: "O modelo torna-se mais explicável à medida que melhora?"
   - Possibilita determinar se a estrutura do modelo está a evoluir significativamente entre rondas

A abordagem adotada garante a consistência dos dados entre rondas, permitindo assim isolar e analisar exclusivamente a evolução da explicabilidade do modelo sem interferências de variações nos dados.

## 🔄 Processo de Desenvolvimento

1. **Preparação da Infraestrutura**
   - Configuração do ambiente Docker
   - Implementação do servidor Flower
   - Desenvolvimento dos clientes

2. **Implementação do Modelo**
   - Escolha e configuração da árvore de decisão
   - Adaptação para o contexto federado
   - Implementação das métricas de avaliação

3. **Sistema de Relatórios**
   - Desenvolvimento de templates HTML
   - Geração de visualizações
   - Armazenamento de métricas em JSON

4. **Visualização da Explicabilidade**
   - Criação de gráficos específicos para cada ronda
   - Implementação do gráfico de evolução da explicabilidade
   - Integração no relatório final HTML

5. **Otimização e Testes**
   - Ajuste de hiperparâmetros
   - Testes de comunicação
   - Validação dos resultados

## 📈 Trabalho Futuro

### Melhorias Propostas
1. **Modelo**
   - Implementar ensemble de árvores (Random Forest)
   - Otimização automática de hiperparâmetros
   - Suporte para mais tipos de modelos

2. **Infraestrutura**
   - Escalabilidade para mais clientes
   - Sistema de recuperação de falhas
   - Monitoramento em tempo real

3. **Privacidade**
   - Implementar técnicas de differential privacy
   - Adicionar criptografia na comunicação
   - Auditoria de privacidade

4. **Interface**
   - Dashboard interativo em tempo real
   - Comparação visual entre clientes
   - Exportação de relatórios em vários formatos

5. **Análise de Explicabilidade**
   - Métricas quantitativas de convergência da explicabilidade
   - Visualizações comparativas interativas entre rondas
   - Análise automática de padrões evolutivos nas árvores
   - Métricas de correlação entre explicabilidade e desempenho

### Próximos Passos
1. Avaliar o impacto do número de rounds no desempenho
2. Implementar validação cruzada federada
3. Desenvolver métricas de qualidade da federação
4. Criar benchmarks com outros modelos
5. Análise aprofundada da evolução da explicabilidade

## 🔍 Conclusões

O projeto demonstrou a viabilidade do treino federado para previsão de preços de imóveis, mantendo a privacidade dos dados e fornecendo resultados interpretáveis. Os RMSEs obtidos (entre 61k e 66k dólares) indicam que o modelo consegue fazer previsões razoáveis, com uma margem de erro aceitável para o mercado imobiliário.

A abordagem federada mostrou-se eficaz para:
- Manter a privacidade dos dados
- Permitir treino distribuído
- Gerar modelos interpretáveis
- Fornecer insights através de visualizações

A métrica de similaridade estrutural (0.95-0.97) demonstrou:
- Alta consistência nas estruturas das árvores entre rondas
- Convergência efetiva do modelo
- Estabilidade nas regras de decisão

O sistema de tracking da evolução da explicabilidade permitiu:
- Acompanhar a evolução do modelo em cada ronda
- Identificar padrões de aprendizagem consistentes
- Verificar a estabilidade das regras de decisão
- Correlacionar mudanças estruturais com métricas de desempenho

Os resultados mostram que o modelo federado é capaz de:
- Manter consistência entre clientes
- Alcançar previsões precisas
- Preservar a privacidade dos dados
- Fornecer interpretabilidade clara

## 📚 Referências

- [Flower Framework](https://flower.dev/)
- [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [California Housing Dataset](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)

## 🧪 Motivação para a Implementação da Métrica de Similaridade Estrutural

Durante a análise dos resultados iniciais do modelo federado, identificou-se uma limitação importante no processo de análise da evolução da explicabilidade:

### Desafio Identificado
- Apesar de métricas como importância das características e complexidade da árvore estarem disponíveis, estas não capturavam adequadamente mudanças subtis na estrutura das árvores de decisão
- Verificou-se uma estabilidade aparente nestas métricas entre rondas, mas não era claro se o modelo estava realmente a convergir ou se as métricas eram insuficientes para capturar a evolução
- A análise manual das estruturas das árvores mostrava algumas diferenças nas regras de decisão que não se refletiam nas métricas existentes
- Faltava uma forma objetiva e quantitativa de determinar o grau de evolução estrutural entre rondas

### Abordagem Conceptual
A solução desenvolvida baseou-se no princípio de que:
- Duas árvores estruturalmente similares devem produzir previsões semelhantes para os mesmos dados
- Árvores com estruturas diferentes produzirão previsões divergentes, mesmo que tenham métricas de importância similares
- A magnitude da diferença nas previsões pode ser utilizada como proxy para quantificar a diferença estrutural

### Vantagens da Nova Métrica
1. **Objetividade**: Substitui avaliações subjetivas por uma métrica numérica precisa
2. **Sensibilidade**: Capaz de detetar mudanças estruturais subtis que não afetam significativamente a importância das características
3. **Interpretabilidade**: Escala normalizada de 0 a 1 facilita a compreensão imediata do grau de mudança
4. **Complementaridade**: Funciona em conjunto com as métricas existentes para uma análise mais completa
5. **Aplicabilidade**: Pode ser utilizada para comparar qualquer par de árvores, independentemente da sua profundidade ou complexidade

### Impacto na Análise
A métrica de similaridade estrutural permite:
- Detetar quando o modelo realmente converge estruturalmente
- Identificar quais rondas produzem as mudanças mais significativas no modelo
- Correlacionar mudanças estruturais com melhorias de desempenho
- Avaliar se a agregação federada está a produzir evolução real ou apenas flutuações mínimas
- Tomar decisões mais informadas sobre quando terminar o processo de treino baseado na estabilização real do modelo

Esta nova métrica preenche uma lacuna crítica na avaliação da explicabilidade, possibilitando uma compreensão mais profunda e quantitativa da evolução do modelo ao longo do treino federado. 