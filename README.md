# Projeto de Aprendizagem Federada e Explicabilidade para Previsão de Preços Imobiliários

Este projeto explora a implementação de modelos de aprendizagem federada (Federated Learning - FL) utilizando o framework Flower para prever os preços de imóveis na Califórnia. O foco principal reside não apenas na previsão, mas também na análise da explicabilidade (Explainability) dos modelos treinados de forma federada.

## Objetivo

Investigar e comparar estratégias para construir modelos interpretáveis em FL, avaliando benefícios, desafios e desempenho. O objetivo é compreender como melhorar a estabilidade dos modelos, reduzir discrepâncias entre clientes e otimizar a interpretabilidade sem comprometer a performance preditiva, sempre preservando a privacidade dos dados.

## Abreviaturas das Abordagens
- **ADF**: Árvore de Decisão Federada (DecisionTreeRegressor, scikit-learn)
- **RLFE**: Regressão Linear Federada Explicável (PyTorch, LIME/SHAP)

## Abordagens Exploradas

1. **ADF:**
   - Interpretabilidade nativa.
   - Importância das features e estrutura da árvore analisadas localmente e agregadas.
   - Implementação: `flower_docker_v2_CH - DecisionTreeRegressor`.
2. **RLFE:**
   - Modelo linear treinado federadamente com Adam.
   - Explicabilidade via LIME e SHAP (gráficos por ronda e evolução global).
   - Implementação: `flower_docker_v2_CH - ExpCli1Cli2`.

Ambas as abordagens usam Docker e um servidor Flower centralizado. O particionamento dos dados é feito com seed fixa, garantindo partições distintas e reprodutíveis para cada cliente.

## Novo Dataset

O novo dataset (`ds_testes_iniciais.csv`) foi criado para facilitar o desenvolvimento e teste das abordagens federadas.
- Contém múltiplas features por amostra, valores ausentes representados por -1.
- É necessário pré-processamento: substituir -1 por NaN, remover colunas não relevantes, confirmar a coluna target (`attack`) e normalizar as features.
- Recomenda-se usar este dataset para debugging e validação inicial antes de escalar para o dataset completo.

## Estrutura Modular RLFE

- A abordagem RLFE (Regressão Linear Federada Explicável) está a ser migrada para uma estrutura modular e escalável.
- Cada cliente RLFE encontra-se em `RFLE/client/` e é parametrizado por `cid` (começando em 1) e `num_clients`.
- O particionamento do dataset é feito "on-the-fly" por cada cliente, garantindo flexibilidade e reprodutibilidade.
- Os outputs de cada cliente são guardados em subpastas dedicadas dentro de `reports/` e `results/`.
- O arranque dos clientes será automatizado via `docker-compose.yml`, que cria N containers de cliente conforme variável `NUM_CLIENTS`.
- Volumes Docker serão usados para persistência dos resultados e partilha do dataset.
- O pipeline de treino federado, avaliação e explicabilidade será integrado no script do cliente após validação da estrutura base.

### Validação e Limpeza de Dados no Cliente RLFE

Após o carregamento do dataset, o pipeline realiza automaticamente:
- **Remoção de colunas constantes:** Colunas sem variância (mesmo valor para todas as amostras) são eliminadas, pois não contribuem para o modelo preditivo.
- **Remoção de colunas de identificação/tempo:** Colunas como `índice`, `_time` e `imeisv` são removidas, pois não têm valor preditivo direto e podem introduzir viés.
- **Tratamento de valores em falta:** Colunas com valores ausentes são preenchidas pela mediana da coluna, tornando o pipeline mais robusto a outliers.
- **Codificação de variáveis categóricas:** Features categóricas são convertidas automaticamente para variáveis dummy (one-hot encoding), garantindo compatibilidade com modelos lineares.

Estas etapas garantem que apenas features relevantes e informativas são usadas no treino e avaliação do modelo federado.

## Execução Escalável de Clientes RLFE

Para garantir que cada container cliente recebe o seu argumento `--cid` único ao usar Docker Compose, utiliza-se o script `generate_compose.py`, que gera automaticamente um ficheiro `docker-compose.generated.yml` com um serviço distinto para cada cliente.

### Passos para execução distribuída:

1. **Escolher o número de clientes**
   - Exemplo: para 4 clientes

2. **Gerar o ficheiro docker-compose**
   ```bash
   python generate_compose.py 4
   ```
   Isto cria o ficheiro `docker-compose.generated.yml` com 4 serviços, cada um com o seu `--cid` (de 1 a 4).

3. **Levantar os clientes federados**
   ```bash
   docker-compose -f docker-compose.generated.yml up --build
   ```
   Cada container será iniciado com o seu `cid` correto e partilhará volumes para reports, results e dataset.

### Notas:
- O script `generate_compose.py` está na raiz da pasta `RFLE/`.
- O volume `DatasetIOT/` é montado como read-only; `reports/` e `results/` são persistentes.
- Para personalizar volumes, nomes de container ou adicionar serviços extra (ex: servidor FL), basta ajustar o template no script.
- O pipeline federado será integrado no cliente após validação da infraestrutura.

## Exemplo de Execução Local

```bash
python client.py --cid=1 --num_clients=4
```

## Próximos Passos
- Finalizar a configuração Docker (Dockerfile e docker-compose.yml)
- Integrar pipeline de treino federado e explicabilidade no cliente RLFE
- Testar execução distribuída e outputs persistentes
- Atualizar documentação após validação

## Estado Atual do Projeto

- Implementação das abordagens federadas concluída para o dataset California Housing.
- Transição em curso para o novo dataset, com recomendações de pré-processamento já definidas.
- Documentação e comparação formal das abordagens em atualização contínua.

## Próximos Passos
- Comparação formal dos resultados entre ADF e RLFE (tabela de métricas, análise de explicabilidade, discussão de limitações).
- Definir e adaptar o novo dataset alvo para a dissertação.
- Desenvolver protocolo de testes e critérios para a comparação robusta das abordagens.
- Desenvolvimento incremental da dissertação (estrutura e escrita das secções principais).

#### Argumentos do Cliente RLFE

- `--cid`: ID do cliente (1-indexed)
- `--num_clients`: número de clientes a executar neste teste
- `--num_total_clients`: **número total de clientes para particionamento dos dados** (define em quantas partes o dataset é dividido, mesmo que só corram alguns clientes)
- `--dataset_path`: caminho para o dataset CSV
- `--seed`: seed para reprodutibilidade

O dataset é sempre dividido por `num_total_clients` e cada cliente recebe apenas a sua fração, tornando o teste mais realista e eficiente mesmo com poucos clientes em execução.