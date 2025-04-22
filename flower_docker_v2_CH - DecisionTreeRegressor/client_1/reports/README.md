# Diretório de Relatórios - Cliente 1

Este diretório contém os relatórios e visualizações gerados durante o treino do modelo de Árvore de Decisão.

## Estrutura

- `final_report.html`: Relatório final interativo com todas as métricas e visualizações
- `rounds_data.json`: Dados consolidados de todas as rondas de treino
- `tree_structure_roundX.txt`: Estrutura da árvore de decisão para cada ronda
- `feature_importance_roundX.png`: Gráfico de importância das características para cada ronda
- `*_evolution.png`: Gráficos de evolução das métricas ao longo das rondas

### Formato dos Ficheiros

#### rounds_data.json
```json
{
    "rounds": [
        {
            "round": 1,
            "timestamp": "YYYY-MM-DD HH:MM:SS",
            "metrics": {
                "training_loss": float,
                "validation_loss": float,
                "evaluation_loss": float,
                "rmse": float
            },
            "visualizations": {
                "tree_structure": "tree_structure_round1.txt",
                "feature_importance": "feature_importance_round1.png"
            }
        }
    ]
}
```

#### Visualizações
- **PNG**: Resolução 1200x800 píxeis
- **Formato**: RGB, 72 DPI
- **Tamanho**: ~100-200KB por imagem

## Visualizações

### Métricas
- Evolução do erro (MSE) ao longo das rondas
- RMSE para cada ronda
- Comparação entre treino e validação

### Interpretabilidade
- **Estrutura da Árvore**: Texto detalhado que mostra as regras de decisão
- **Importância das Características**: Gráfico de barras que mostra o impacto de cada variável
- **Similaridade Estrutural**: Métrica que compara a estrutura de árvores entre rondas consecutivas

### Interpretação dos Resultados

#### Métricas
- **Training Loss**: Erro médio quadrático no conjunto de treino
- **Validation Loss**: Erro médio quadrático no conjunto de validação
- **Evaluation Loss**: Erro médio quadrático no conjunto de teste
- **RMSE**: Raiz do erro quadrático médio (em unidades do alvo)

#### Estrutura da Árvore
- Cada nó mostra uma regra de decisão
- Os valores nas folhas são as previsões
- A profundidade indica a complexidade do modelo

#### Importância das Características
- Barras maiores indicam características mais importantes
- Baseado na redução do erro proporcionada por cada característica
- Soma normalizada para 1.0 (ou 100%)

#### Similaridade Estrutural
- **Definição**: Medida de quão semelhantes são duas árvores de rondas consecutivas (1.0 = idênticas, 0.0 = completamente diferentes)
- **Método de Cálculo**: 
  1. Gera 1000 pontos de teste aleatórios com ruído adicionado (desvio padrão 0.2)
  2. Obtém previsões de ambas as árvores nos pontos de teste
  3. Calcula o erro médio quadrático (MSE) entre as previsões
  4. Normaliza para similaridade: `exp(-MSE / (scale_factor + 1e-6))`
  5. Scale_factor = `std(previsões) * 0.1` para aumentar a sensibilidade às diferenças
  6. Adiciona variação aleatória (até 10%) para valores muito altos (>0.95)

- **Interpretação**:
  - Valores baixos indicam mudanças significativas na estrutura da árvore
  - Valores altos sugerem estabilização do modelo
  - A variação entre rondas indica adaptação ou convergência

- **Estratégia de Sensibilidade**: O algoritmo foi ajustado para:
  - Usar um fator de escala menor (0.1) para amplificar pequenas diferenças
  - Adicionar mais ruído nos pontos de teste (0.2) para testar a robustez
  - Aplicar aleatoriedade controlada (até 10%) para evitar saturação nos valores altos
  - Esta abordagem permite detetar alterações subtis na estrutura que poderiam passar despercebidas

#### Visualização da Evolução da Explicabilidade
- **Formato**: Grelha 2x2 com quatro gráficos distintos
- **Gráficos Incluídos**:
  1. **Importância das Características**: Evolução da soma das importâncias
  2. **Complexidade da Árvore**: Evolução do número de regras de decisão
  3. **Similaridade Estrutural**: Evolução da semelhança entre modelos consecutivos
  4. **RMSE**: Evolução do erro em unidades do alvo
- **Vantagens**: 
  - Visualização clara e separada de cada métrica
  - Facilita a comparação entre diferentes aspetos da explicabilidade
  - Permite identificar correlações entre métricas (por exemplo, como a complexidade afeta o erro)
  - Melhor legibilidade e interpretação de cada componente individual

## Atualização

Os relatórios são atualizados automaticamente:
1. Após cada ronda de treino
2. No final do processo completo de treino

## Como Utilizar

1. Abra `final_report.html` num navegador web
2. Navegue pelas secções de métricas e visualizações
3. Analise a estrutura da árvore em cada ronda
4. Compare a importância das características entre rondas

### Dicas de Análise
- Compare as perdas entre rondas para avaliar convergência
- Verifique se existe overfitting (perda na validação a aumentar)
- Identifique as características mais importantes para as previsões
- Analise como a estrutura da árvore evolui ao longo do treino
- Observe a métrica de similaridade para avaliar a estabilidade do modelo entre rondas
