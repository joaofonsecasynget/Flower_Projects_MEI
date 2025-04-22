# Guia de Explicabilidade

Este guia detalha as técnicas de explicabilidade implementadas no projeto.

## LIME (Local Interpretable Model-agnostic Explanations)

### Visão Geral
LIME é uma técnica que explica previsões individuais do modelo através da aproximação local do comportamento do modelo.

### Implementação
- Utiliza a biblioteca `lime` versão 0.2.0+
- Gera explicações para cada round de treino
- Foca nas 5 features mais importantes

### Interpretação
1. **Barras Coloridas**
   - Verde: Contribuição positiva para o preço
   - Vermelho: Contribuição negativa para o preço
   - Comprimento: Magnitude da contribuição

2. **Features**
   - Nome da feature
   - Valor atual da feature
   - Contribuição para a previsão

3. **Exemplo de Leitura**
```
Feature: median_income = 8.5
Contribuição: +$50,000
Interpretação: Um aumento na renda mediana está associado a um aumento no preço
```

## SHAP (SHapley Additive exPlanations)

### Visão Geral
SHAP utiliza teoria dos jogos para calcular a importância de cada feature nas previsões do modelo.

### Implementação
- Utiliza a biblioteca `shap` versão 0.41.0+
- Gera sumários globais do modelo
- Analisa todas as features simultaneamente

### Interpretação
1. **Gráfico de Sumário**
   - Eixo Y: Features ordenadas por importância
   - Eixo X: Valores SHAP (impacto na previsão)
   - Cores: Valor da feature (vermelho = alto, azul = baixo)

2. **Valores SHAP**
   - Positivos: Aumentam o preço previsto
   - Negativos: Diminuem o preço previsto
   - Magnitude: Importância da feature

3. **Exemplo de Leitura**
```
Feature: ocean_proximity
Valor SHAP: 0.5
Cor: Vermelho
Interpretação: Proximidade ao oceano tem forte impacto positivo no preço
```

## Geração de Relatórios

### Processo
1. Coleta de dados durante o treino
2. Geração de explicações LIME e SHAP
3. Criação de visualizações
4. Compilação do relatório final

### Arquivos Gerados
- `lime_explanation_roundX.png`
- `shap_summary_roundX.png`
- Métricas em `rounds_data.json`
- Relatório final em HTML

## Melhores Práticas

### Análise de Resultados
1. Compare explicações entre rounds
2. Verifique consistência das features importantes
3. Identifique padrões nas contribuições
4. Valide com conhecimento do domínio

### Limitações
- LIME: Explicações locais podem variar
- SHAP: Computacionalmente intensivo
- Ambos: Requerem interpretação cuidadosa

## Referências
- [LIME Paper](https://arxiv.org/abs/1602.04938)
- [SHAP Paper](https://arxiv.org/abs/1705.07874)
- [Documentação LIME](https://lime-ml.readthedocs.io/)
- [Documentação SHAP](https://shap.readthedocs.io/) 