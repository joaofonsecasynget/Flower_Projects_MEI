# Flower Docker - Treino Federado

Este projeto implementa um sistema de treino federado usando o framework Flower, com dois clientes a treinar modelos de regressão linear de forma distribuída, incluindo capacidades de explicabilidade através de LIME e SHAP.

## Estrutura do Projeto

- `client_1/`: Código e dados do Cliente 1
- `client_2/`: Código e dados do Cliente 2
- `server/`: Código do servidor Flower
- `data/`: Conjuntos de dados para treino
- `docker/`: Ficheiros Docker e configurações
- `docs/`: Documentação detalhada do projeto

## Requisitos

### Versões das Dependências
- Python 3.8+
- Docker 20.10+
- Docker Compose 2.0+
- Flower 1.5.0
- PyTorch 1.13.0+
- Pandas 1.3.0+
- Scikit-learn 0.24.0+
- LIME 0.2.0+
- SHAP 0.41.0+

## Conjunto de Dados

O projeto utiliza o conjunto de dados California Housing com as seguintes características:
- **Features**: longitude, latitude, housing_median_age, total_rooms, total_bedrooms, population, households, median_income, ocean_proximity
- **Target**: median_house_value
- **Tamanho**: 20.640 amostras
- **Divisão**: 80% treino, 20% teste para cada cliente

## Instalação

1. Clone o repositório:
```bash
git clone https://github.com/seu-utilizador/flower-docker.git
cd flower-docker
```

2. Construa as imagens Docker:
```bash
docker-compose build
```

## Configuração

### Parâmetros Principais
- Número de rondas: 5 (configurável em `server/server.py`)
- Batch size: 32 (configurável nos clientes)
- Learning rate: 0.001 (otimizador Adam)
- Validação: 10% dos dados de treino

## Utilização

1. Inicie os contentores:
```bash
docker-compose up
```

2. Monitorize o treino:
- Aceda aos logs dos contentores para acompanhar o progresso
- Verifique os relatórios gerados em `client_X/reports/`

3. Após o treino:
- Analise os relatórios finais em HTML
- Verifique as métricas e visualizações geradas
- Consulte os modelos guardados em `client_X/results/`

### Resultados Esperados
- Loss inicial: ~5.5e7
- RMSE final: ~235.000
- Tempo médio por ronda: 6-7 segundos

## Funcionalidades

### Treino Federado
- Treino distribuído com Flower
- Agregação de modelos no servidor
- Métricas de desempenho por ronda

### Explicabilidade
- Visualizações LIME e SHAP persistentes
- Instâncias fixas para explicações LIME consistentes entre rondas
- Acompanhamento da evolução das explicações ao longo do treino
- Análise de importância de features
- Relatórios detalhados por ronda

#### Sistema de Explicabilidade
O projeto implementa um sistema robusto de explicabilidade que:
- Inicializa o explainer LIME apenas uma vez por cliente
- Seleciona e mantém uma instância fixa para explicações LIME em todas as rondas
- Permite comparar como as explicações para a mesma instância evoluem ao longo do treino
- Gera visualizações SHAP para análise global do modelo
- Produz relatórios HTML com a evolução das métricas e explicações

### Monitorização
- Logs detalhados do treino
- Métricas de desempenho
- Visualizações da evolução do modelo

## Resolução de Problemas

### Problemas Comuns
1. **Erro de ligação com o servidor**:
   - Verifique se a porta 9091 está livre
   - Confirme se o servidor iniciou corretamente

2. **Erros de memória**:
   - Ajuste o batch size nos clientes
   - Verifique a disponibilidade de RAM

3. **Falha na geração de relatórios**:
   - Verifique as permissões das pastas
   - Confirme se há espaço em disco

4. **Problemas com explicabilidade**:
   - Verifique a inicialização do LIME explainer nos logs do cliente
   - Confirme se a instância fixa está a ser selecionada corretamente
   - Verifique se o contador de rondas está a incrementar adequadamente

## Documentação

Para mais detalhes sobre o funcionamento do sistema, consulte:
- [Documentação do Flower](https://flower.dev/docs/)
- [Guia de Explicabilidade](docs/explainability.md)
- [Configuração do Docker](docs/docker-setup.md)
