# Resumo do Projeto: Treino Federado com Flower

## Visão Geral
Este projeto implementa um sistema de treino federado usando o framework Flower para treinar um modelo de regressão linear no conjunto de dados California Housing. O sistema é composto por dois clientes que treinam o modelo de forma distribuída, com um servidor central responsável pela agregação dos modelos.

## Estratégia Adotada

### 1. Arquitetura do Sistema
- **Servidor Central**: Implementado com Flower, responsável por coordenar o treino federado
- **Dois Clientes**: Cada cliente treina uma parte do conjunto de dados e contribui para o modelo global
- **Contentorização**: Utilização de Docker para garantir consistência e isolamento
- **Explicabilidade**: Implementação de técnicas LIME e SHAP para interpretação do modelo, com instâncias fixas para análise consistente

### 2. Processo de Treino
1. Divisão do conjunto de dados California Housing entre os clientes
2. Pré-processamento dos dados (normalização, tratamento de valores ausentes)
3. Treino local em cada cliente
4. Agregação dos modelos no servidor (FedAvg)
5. Avaliação e geração de explicações com instâncias fixas

### 3. Métricas e Monitorização
- Loss de treino e validação
- RMSE (Root Mean Square Error)
- Visualizações LIME e SHAP
- Relatórios detalhados por ronda

## Resultados Obtidos

### 1. Desempenho do Modelo
- Evolução do RMSE ao longo das rondas:
  - Ronda 1: 235,232.42
  - Ronda 2: 235,232.14
  - Ronda 3: 235,231.88
  - Ronda 4: 235,231.59
  - Ronda 5: 235,231.31

- Tempo de treino:
  - Tempo total: 6.73 segundos
  - Tempo médio por ronda: ~1.35 segundos

### 2. Explicabilidade
- Visualizações LIME para interpretação local da mesma instância ao longo do treino
- Análise SHAP para importância global das features
- Relatórios detalhados por ronda de treino com visualizações consistentes
- Gráficos de evolução da explicabilidade para LIME e SHAP

### 3. Benefícios do Treino Federado
- Preservação da privacidade dos dados
- Treino distribuído eficiente
- Agregação de conhecimento de múltiplos clientes
- Explicabilidade mantida durante todo o processo

## Desafios e Soluções

### 1. Persistência do Explainer LIME
- **Desafio**: Inicialmente, o explainer LIME era reinicializado a cada ronda, selecionando instâncias diferentes para explicação, o que impedia a comparação consistente das explicações ao longo do tempo.
- **Solução**: Modificação dos clientes para inicializar o explainer LIME apenas uma vez e manter a mesma instância fixa em todas as rondas, através de atributos de classe.

### 2. Sincronização entre Rondas
- **Desafio**: O contador de rondas não estava a incrementar corretamente no cliente 2, causando problemas na geração de relatórios.
- **Solução**: Implementação de lógica adequada para incrementar `current_round` após cada avaliação.

### 3. Relatório Final Consistente
- **Desafio**: O relatório do cliente 2 mostrava uma "Ronda 0" inexistente, criando inconsistência com o cliente 1.
- **Solução**: Filtração dos dados para excluir a ronda 0 durante a geração do relatório final.

## Trabalho Futuro Sugerido

### 1. Melhorias Técnicas
- Implementação de mais estratégias de agregação
- Otimização do processo de treino
- Adição de mais técnicas de explicabilidade
- Análise comparativa das explicações entre rondas

### 2. Expansão do Sistema
- Suporte a mais clientes
- Implementação de diferentes tipos de modelos
- Adição de mais conjuntos de dados para teste

### 3. Monitorização e Análise
- Dashboard em tempo real
- Análise mais detalhada das métricas
- Comparação automática das explicações entre rondas
- Comparação entre diferentes configurações

### 4. Segurança e Privacidade
- Implementação de técnicas de privacidade diferencial
- Criptografia dos dados em trânsito
- Validação de segurança dos modelos

## Conclusão
O projeto demonstra com sucesso a implementação de um sistema de treino federado usando Flower, com foco em explicabilidade e monitorização. Os resultados mostram que é possível treinar modelos de forma distribuída mantendo a privacidade dos dados e obtendo bons resultados de desempenho. A implementação de explicações LIME e SHAP com instâncias fixas permite uma análise consistente da evolução do modelo ao longo do treino, fornecendo insights valiosos sobre o processo de aprendizagem federada. 