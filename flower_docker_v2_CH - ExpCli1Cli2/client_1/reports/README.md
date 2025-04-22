# Diretório de Relatórios - Cliente 1

Este diretório contém os relatórios e visualizações gerados durante o treino do modelo do Cliente 1.

## Estrutura

- `final_report.html`: Relatório final interativo com todas as métricas e explicações
- `rounds_data.json`: Dados consolidados de todas as rounds de treino
- `lime_explanation_roundX.png`: Explicações LIME para cada round
- `shap_summary_roundX.png`: Sumários SHAP para cada round
- `training_loss_evolution.png`: Gráfico da evolução do training loss
- `validation_loss_evolution.png`: Gráfico da evolução do validation loss
- `evaluation_loss_evolution.png`: Gráfico da evolução do evaluation loss

### Formato dos Arquivos

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
                "lime": "lime_explanation_round1.png",
                "shap": "shap_summary_round1.png"
            }
        }
    ]
}
```

#### Visualizações
- **PNG**: Resolução 1200x800 pixels
- **Formato**: RGB, 72 DPI
- **Tamanho**: ~100-200KB por imagem

## Visualizações

### Métricas
- Gráficos separados para cada métrica (training, validation, evaluation loss)
- Evolução temporal ao longo das rounds
- Grid e legendas para fácil interpretação

### Explicabilidade
- **LIME**: Explicações locais com importância das features
- **SHAP**: Impacto global das features nas previsões

### Interpretação dos Resultados

#### Métricas
- **Training Loss**: Erro médio quadrático no conjunto de treino
- **Validation Loss**: Erro médio quadrático no conjunto de validação
- **Evaluation Loss**: Erro médio quadrático no conjunto de teste
- **RMSE**: Raiz do erro quadrático médio (em unidades do target)

#### Visualizações LIME
- Barras verdes: Impacto positivo no preço
- Barras vermelhas: Impacto negativo no preço
- Tamanho das barras: Magnitude do impacto

#### Visualizações SHAP
- Cores quentes (vermelho): Valores altos da feature
- Cores frias (azul): Valores baixos da feature
- Largura: Distribuição dos valores SHAP

## Atualização

Os relatórios são atualizados automaticamente:
1. Após cada round de treino
2. Ao final do processo completo de treino

## Como Usar

1. Abra `final_report.html` em um navegador web
2. Navegue pelas seções de métricas e explicabilidade
3. Analise as visualizações individuais em formato PNG
4. Consulte `rounds_data.json` para dados brutos

### Dicas de Análise
- Compare as losses entre rounds para avaliar convergência
- Verifique se há overfitting (validation loss aumentando)
- Analise quais features têm maior impacto nas previsões
- Observe padrões nas explicações LIME e SHAP
