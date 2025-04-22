# Interpretabilidade do Modelo

Este documento descreve a abordagem de interpretabilidade utilizada no projeto de aprendizagem federada usando Árvores de Decisão.

## Visão Geral

O projeto utiliza Árvores de Decisão como modelo preditivo, que são naturalmente interpretáveis. Isso significa que podemos entender facilmente como o modelo toma suas decisões sem necessidade de ferramentas adicionais de explicabilidade.

## Características da Interpretabilidade

1. **Visualização da Árvore**
   - Estrutura hierárquica clara
   - Regras de decisão explícitas em cada nó
   - Caminhos de decisão facilmente rastreáveis

2. **Importância das Características**
   - Ranking direto das características mais influentes
   - Medidas baseadas na redução da impureza
   - Visualização através de gráficos de barras

3. **Regras de Decisão**
   - Conjunto de regras IF-THEN claras e compreensíveis
   - Fácil tradução para linguagem de negócios
   - Insights acionáveis para stakeholders

## Geração de Relatórios

Para cada rodada de treinamento, são gerados:
1. Visualização da estrutura da árvore
2. Gráfico de importância das características
3. Métricas de desempenho (RMSE, MSE)

O relatório final consolida todas as visualizações e métricas, permitindo acompanhar a evolução do modelo ao longo das rodadas.

## Vantagens da Abordagem

- Transparência total no processo de decisão
- Sem necessidade de ferramentas externas de explicabilidade
- Facilidade de interpretação por não-especialistas
- Insights diretos para o domínio do problema

## Referências

1. [Documentação do Scikit-learn - Decision Trees](https://scikit-learn.org/stable/modules/tree.html)
2. [Interpretable Machine Learning](https://christophm.github.io/interpretable-ml-book/tree.html)
3. [Feature Importance in Decision Trees](https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html) 