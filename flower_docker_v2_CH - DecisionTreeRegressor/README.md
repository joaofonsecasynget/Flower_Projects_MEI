# Flower Docker - Federated Learning with Decision Trees 🌳
# Flower Docker - Treino Federado com Árvores de Decisão 🌳

This project implements a federated learning system using the Flower framework, with two clients training decision tree models in a distributed way for house price prediction.

Este projeto implementa um sistema de treino federado com o framework Flower, onde dois clientes treinam modelos de árvores de decisão de forma distribuída para a previsão de preços de imóveis.

## Project Structure | Estrutura do Projeto

```
.
├── client_1/           # Client 1 code and data | Código e dados do Cliente 1
├── client_2/           # Client 2 code and data | Código e dados do Cliente 2
├── server/            # Flower server code | Código do servidor Flower
├── data/             # Training datasets | Conjuntos de dados para treino
├── docker/           # Docker files and configs | Ficheiros Docker e configurações
└── docs/             # Project documentation | Documentação do projeto
```

## Data Distribution | Distribuição dos Dados

O projeto utiliza o conjunto de dados California Housing para treinar os modelos de previsão de preços de imóveis. A distribuição dos dados ocorre da seguinte forma:

### Dataset Base
- **Origem**: California Housing Dataset
- **Conteúdo**: Dados demográficos e habitacionais de diferentes regiões da Califórnia
- **Alvo**: Valor mediano das casas (`median_house_value`)

### Partição dos Dados
- **Cliente 1**: 
  - Utiliza uma amostra do conjunto completo, dividida pela função `load_datasets`
  - Os dados são divididos aleatoriamente utilizando a semente aleatória 42
  - A função cria partições específicas para ambos os clientes
  - A partição é feita de forma que cada cliente receba dados de diferentes regiões

- **Cliente 2**:
  - Utiliza o mesmo conjunto de dados base, mas processado de forma independente
  - Implementa sua própria lógica de carregamento através da função `load_data`
  - Mantém a consistência da divisão com a mesma semente aleatória 42

### Pré-processamento
- **Normalização**: Ambos os clientes normalizam os dados utilizando `StandardScaler`
- **Valores ausentes**: Preenchidos com a média da coluna
- **Variáveis categóricas**: A coluna `ocean_proximity` é convertida para valores numéricos
- **Divisões**:
  - Treino: 80% dos dados
  - Validação: 10% dos dados de treino
  - Teste: 20% dos dados

### Importante
Os dados utilizados em cada ronda de treino são os mesmos ao longo de todo o processo federado. Isto permite uma análise consistente da evolução da explicabilidade, pois as mudanças observadas refletem o aprendizado do modelo e não variações nos dados.

## Decision Tree Model | Modelo de Árvore de Decisão

### Model Configuration | Configuração do Modelo
- **Type | Tipo**: DecisionTreeRegressor (scikit-learn)
- **Max Depth | Profundidade Máxima**: 10
- **Random State**: 42
- **Training Rounds | Rondas de Treino**: 5
- **Validation Split | Divisão de Validação**: 10%
- **Metrics | Métricas**: MSE, RMSE

### Features | Características
- longitude
- latitude
- housing_median_age | idade média da habitação
- total_rooms | total de divisões
- total_bedrooms | total de quartos
- population | população
- households | agregados familiares
- median_income | rendimento médio
- ocean_proximity | proximidade ao oceano

### Model Performance | Desempenho do Modelo
- **Training Loss | Perda no Treino**: ~1.84B
- **Validation Loss | Perda na Validação**: ~3.98B (Cliente 1), ~4.12B (Cliente 2)
- **Evaluation Loss | Perda na Avaliação**: ~4.39B (Cliente 1), ~3.77B (Cliente 2)
- **RMSE**: ~66,234.26 (Cliente 1), ~61,409.84 (Cliente 2)

## Explainability Evolution | Evolução da Explicabilidade

As árvores de decisão oferecem vantagens naturais de interpretabilidade. Este projeto implementa mecanismos para acompanhar e analisar a evolução desta explicabilidade ao longo das rondas de treino federado.

### Artefactos de Explicabilidade
Em cada ronda de treino, o sistema gera automaticamente:

1. **Estrutura da Árvore**: 
   - Ficheiro de texto com a representação completa da árvore (`tree_structure_roundX.txt`)
   - Mostra as regras de decisão em cada nó, valores de divisão e predições nas folhas
   - Permite analisar como as regras evoluem entre rondas

2. **Importância das Características**:
   - Gráfico de barras mostrando a importância relativa de cada característica (`feature_importance_roundX.png`)
   - Permite identificar quais características ganham ou perdem relevância entre rondas
   - Facilita a comparação da evolução do modelo em termos de quais variáveis são mais determinantes

3. **Métricas de Desempenho**:
   - Valores de MSE e RMSE calculados para cada conjunto (treino, validação, teste)
   - Armazenados em JSON para facilitar análises comparativas
   - Permitem verificar se melhorias de desempenho correlacionam-se com mudanças na estrutura do modelo

4. **Gráfico da Evolução da Explicabilidade**:
   - Visualização que combina a evolução da importância das características e da complexidade da árvore
   - Integra também a evolução do RMSE para correlacionar desempenho com explicabilidade
   - Formato: `explainability_evolution.png`

5. **Métrica de Similaridade Estrutural**:
   - Nova métrica que quantifica numericamente a similaridade entre árvores de diferentes rondas
   - Valores entre 0 e 1, onde 1 indica árvores idênticas e 0 indica árvores completamente diferentes
   - Calculada com base na concordância das previsões em pontos aleatórios
   - Permite identificar quando ocorrem mudanças significativas na estrutura do modelo

6. **Relatório Final Integrado**:
   - HTML agregando todos os artefactos anteriores (`final_report.html`)
   - Permite visualizar lado a lado a evolução ao longo das rondas
   - Facilita a identificação de padrões e tendências na evolução do modelo

### Análise da Evolução
O sistema permite analisar vários aspetos da evolução da explicabilidade:

1. **Convergência do Modelo**:
   - Observar se a estrutura da árvore estabiliza ao longo das rondas
   - Analisar se os nós de decisão principais permanecem consistentes
   - Utilizar a métrica de similaridade estrutural para quantificar a estabilidade

2. **Evolução da Importância**:
   - Verificar se as características mais importantes se mantêm ou mudam
   - Identificar características que ganham importância ao longo do treino

3. **Estabilidade das Regras**:
   - Analisar se as regras de decisão se tornam mais específicas ou generalistas
   - Verificar se os valores de corte dos nós convergem para valores semelhantes

4. **Relação com Desempenho**:
   - Correlacionar mudanças na estrutura com melhorias nas métricas
   - Identificar padrões que indicam maior potencial de generalização
   - Avaliar se a estabilidade (alta similaridade) corresponde a melhor desempenho

### Visualização e Comparação
- Os relatórios são gerados de forma a facilitar a comparação visual entre rondas
- As visualizações mantêm escalas consistentes para permitir análises diretas
- Os dados brutos são disponibilizados em formatos acessíveis (JSON, TXT) para análises personalizadas

## Motivação para a Métrica de Similaridade Estrutural

Durante a análise do modelo, observou-se um desafio na quantificação da evolução estrutural das árvores entre rondas de treino. Enquanto métricas como importância das características e desempenho (RMSE) eram facilmente quantificáveis, a evolução estrutural da árvore permanecia difícil de medir objetivamente.

### Problema Identificado
- Observou-se pouca ou nenhuma variação na importância das características e complexidade da árvore entre rondas
- Não era claro se esta estabilidade representava convergência precoce do modelo ou limitações na agregação das árvores
- Faltava uma métrica objetiva para determinar se as árvores estavam de facto a mudar estruturalmente

### Solução Desenvolvida
A métrica de similaridade estrutural foi desenvolvida para:
- Quantificar objetivamente as mudanças estruturais entre árvores de rondas consecutivas
- Basear-se no comportamento funcional das árvores (previsões) em vez de comparações diretas da estrutura
- Proporcionar uma escala normalizada (0-1) que facilita a interpretação
- Permitir correlacionar estabilidade estrutural com desempenho preditivo

### Implementação
- Geração de pontos aleatórios no espaço de características
- Comparação das previsões de árvores de rondas consecutivas nos mesmos pontos
- Normalização da diferença utilizando uma função de decaimento exponencial
- Integração com visualizações existentes para facilitar análise holística

Esta métrica permite determinar se a ausência de evolução aparente nas métricas tradicionais se deve a uma real estabilidade do modelo ou a limitações na forma como avaliamos a explicabilidade.

## Requirements | Requisitos

### Dependencies | Dependências
- Python 3.12+
- Docker 20.10+
- Docker Compose 2.0+
- Flower 1.5.0
- Pandas 2.2.0+
- Scikit-learn 1.4.0+
- Matplotlib 3.8.0+

## Installation | Instalação

1. Clone the repository | Clone o repositório:
```bash
git clone https://github.com/seu-utilizador/flower-docker.git
cd flower-docker
```

2. Build Docker images | Construa as imagens Docker:
```bash
docker-compose build
```

## Usage | Utilização

1. Start containers | Inicie os contentores:
```bash
docker-compose up
```

2. Monitor training | Monitorize o treino:
- Access container logs | Aceda aos registos dos contentores
- Check reports in `client_X/reports/` | Consulte os relatórios em `client_X/reports/`

3. Analyze results | Analise os resultados:
- View tree structure | Visualize a estrutura da árvore
- Check feature importance | Verifique a importância das características
- Compare metrics between rounds | Compare as métricas entre rondas
- Analyze explainability evolution | Analise a evolução da explicabilidade

### Generated Outputs | Resultados Gerados
For each training round | Para cada ronda de treino:
- Tree structure | Estrutura da árvore (`tree_structure_roundX.txt`)
- Feature importance | Importância das características (`feature_importance_roundX.png`)
- Performance metrics | Métricas de desempenho (MSE, RMSE)
- Metrics evolution | Evolução das métricas

Final outputs | Resultados finais:
- Explainability evolution chart | Gráfico de evolução da explicabilidade (`explainability_evolution.png`)
- Complete HTML report | Relatório HTML completo (`final_report.html`)
- Metrics data in JSON | Dados das métricas em JSON (`rounds_data.json`)

## Troubleshooting | Resolução de Problemas

### Common Issues | Problemas Comuns
1. **Server connection error | Erro de ligação ao servidor**:
   - Check if port 9091 is free | Verifique se a porta 9091 está livre
   - Confirm server started correctly | Confirme se o servidor iniciou corretamente

2. **Memory errors | Erros de memória**:
   - Adjust tree max depth | Ajuste a profundidade máxima da árvore
   - Check RAM availability | Verifique a disponibilidade de RAM

3. **Report generation failure | Falha na geração de relatórios**:
   - Check folder permissions | Verifique as permissões das pastas
   - Confirm disk space | Confirme o espaço em disco
   - Ensure matplotlib is working | Garanta que o matplotlib está a funcionar corretamente

## Documentation | Documentação

For more details about the system, check:
Para mais detalhes sobre o sistema, consulte:
- [Flower Documentation](https://flower.dev/docs/)
- [Scikit-learn Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
- [Docker Setup](docs/docker-setup.md)
