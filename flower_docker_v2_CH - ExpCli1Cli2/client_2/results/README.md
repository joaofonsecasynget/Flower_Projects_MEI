# 📊 Diretório de Resultados - Cliente 2

Este diretório armazena os resultados do treinamento e avaliação do modelo do Cliente 2.

## 📁 Conteúdo

- `model_client_2.pt`: Modelo PyTorch treinado
- Métricas de avaliação
- Dados de validação cruzada
- Checkpoints do modelo (quando aplicável)

## 🔢 Métricas Armazenadas

- Training Loss
- Validation Loss
- Evaluation Loss
- RMSE (Root Mean Square Error)
- Métricas de Cross-Validation

## 💾 Formato dos Dados

- Modelos: Formato PyTorch (.pt)
- Métricas: JSON
- Logs: Texto plano

## 🔄 Atualização

Os resultados são atualizados:
1. Durante o treinamento (checkpoints)
2. Após cada round de federated learning
3. Ao final do treinamento completo

## 📈 Análise

Para analisar os resultados:
1. Carregue o modelo usando PyTorch
2. Consulte as métricas nos arquivos JSON
3. Verifique os logs para informações detalhadas
4. Compare resultados entre rounds
5. Analise as explicações LIME e SHAP para interpretabilidade

## 📝 Observações

- Implementa todas as funcionalidades do Cliente 1
- Inclui métricas avançadas como RMSE
- Oferece análises de explicabilidade com LIME e SHAP
