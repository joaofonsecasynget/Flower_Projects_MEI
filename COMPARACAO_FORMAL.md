# Comparação Formal das Abordagens Federadas

Este documento apresenta um template de tabela comparativa e uma estrutura detalhada para a secção de comparação formal entre as abordagens de aprendizagem federada desenvolvidas no projeto, com foco na implementação **CLFE** utilizando o **dataset IoT**:

- **CLFE**: Classificação Linear Federada com Explicabilidade via LIME e SHAP (PyTorch) utilizando o dataset IoT.
- **ADF**: Árvore de Decisão Federada (DecisionTreeRegressor, scikit-learn) - *Nota: Resultados anteriores eram do dataset California Housing. A comparação direta requer testes com ADF no dataset IoT.*

> **Abreviaturas utilizadas:**
> - **ADF**: Árvore de Decisão Federada (DecisionTreeRegressor, scikit-learn)
> - **CLFE**: Classificação Linear Federada Explicável (PyTorch, LIME/SHAP, Dataset IoT)

---

## 1. Tabela Comparativa de Resultados (Foco CLFE no Dataset IoT)

| Critério / Métrica                                      | CLFE (Dataset IoT) | ADF (Dataset California Housing - Referência Antiga) |
|---------------------------------------------------------|:------------------:|:----------------------------------------------------:|
| **Accuracy Médio (entre clientes)**                     | ~0.99 | N/A (Regressão) |
| **Precision Médio (entre clientes)**                    | ~0.98 | N/A (Regressão) |
| **Recall Médio (entre clientes)**                       | ~0.97 | N/A (Regressão) |
| **F1-Score Médio (entre clientes)**                     | ~0.98 | N/A (Regressão) |
| **Perda de Validação Média**                            | ~0.21 | ~29B (Média dos valores antigos C1: 55B, C2: 4B)   |
| **Tempo de Execução Médio (por ronda)**                 | ~0.08s (fit) + ~0.002s (evaluate) | [Não medido anteriormente] |
| **Estabilidade entre rondas (variação de métricas)**    | Alta (valores consistentes após ronda 3) | [Não analisado anteriormente] |
| **Qualidade da Explicabilidade (Global)**               | Boa (SHAP feature importance) | Boa (Estrutura da árvore) |
| **Qualidade da Explicabilidade (Local)**                | Excelente (LIME/SHAP para instâncias específicas) | Média (Caminho na árvore) |
| **Visualização da Explicabilidade**                     | Gráficos LIME (Top10/Completo), SHAP, Visualizações temporais | Gráfico da árvore, importância das features |
| **Robustez à distribuição dos dados**                   | Alta (desempenho consistente entre clientes) | Variação significativa observada entre clientes |
| **Limitações Observadas**                               | Convergência muito rápida, valores perfeitos no conjunto de teste | Problemas de gráficos (antigo); variação entre clientes |

> **Notas:**
> - Valores baseados nas execuções recentes da CLFE no dataset IoT (`reports/client_X/metrics_history.json`, `reports/client_X/final_report.html`).
> - A coluna ADF serve apenas como referência da implementação anterior com outro dataset. Uma comparação formal exigiria re-executar ADF no dataset IoT.
> - O foco principal é caracterizar o desempenho e a explicabilidade da CLFE no cenário atual.

---

## 2. Estrutura Sugerida para a Secção de Análise (Foco CLFE)

### 4. Análise da Abordagem CLFE com Dataset IoT

#### 4.1. Desempenho Preditivo
- Análise das métricas de classificação (accuracy, precision, recall, F1) e perda de validação (média e por cliente).
- Discussão da convergência ao longo das rondas (baseado nos plots de evolução).
- Estabilidade entre clientes (comparar métricas finais dos diferentes clientes).

#### 4.2. Análise da Explicabilidade (LIME/SHAP)
- Discussão das features mais importantes identificadas pelo LIME (Top 10 e completo) e SHAP.
- Exemplos de interpretação local para instâncias específicas (se relevante).
- Avaliação da utilidade das explicações no contexto do problema IoT.
- Apresentação dos gráficos LIME/SHAP gerados.

#### 4.3. Eficiência e Recursos
- Análise dos tempos de execução (`fit_duration`, `evaluate_duration`) por ronda.
- Discussão do impacto da geração de explicabilidade (apenas na ronda final) no tempo total.

#### 4.4. Limitações e Considerações
- Discussão sobre a adequação do modelo de classificação linear ao dataset IoT (com base nas métricas).
- Observações sobre a heterogeneidade dos dados entre clientes (se aplicável).
- Outras limitações encontradas durante a execução e análise.

#### 4.5. Síntese
- Resumo dos pontos fortes e fracos da abordagem CLFE neste cenário.

---

## 1. Resultados Quantitativos (Contextualização)

A análise quantitativa foca-se agora na abordagem **CLFE (Classificação Linear Federada Explicável)** implementada com **PyTorch** e utilizando o **dataset IoT**. Essa abordagem representa uma evolução da anterior RLFE, após identificarmos que o problema com o dataset IoT é de classificação binária (detecção de ataques) e não de regressão. Resultados anteriores da abordagem ADF (Árvore de Decisão Federada com scikit-learn) pertencem a um contexto diferente (dataset California Housing) e servem apenas como referência histórica.

**Classificação Linear Federada (CLFE) - Dataset IoT**
- **Accuracy médio (clientes):** ~0.99 (excelente para detecção de ataques)
- **Precision médio (clientes):** ~0.98 (pouquíssimos falsos positivos)
- **Recall médio (clientes):** ~0.97 (pouquíssimos falsos negativos)
- **F1-Score médio (clientes):** ~0.98 (harmonia entre precision e recall)
- **Tempos de execução:** Fit (~0.08s/ronda), evaluate (~0.002s/ronda), LIME (~2.5s), SHAP (~12s)

A análise detalhada destes valores, incluindo a sua evolução e variação entre clientes, permite caracterizar o excelente desempenho da CLFE no problema de detecção de ataques IoT.

## 2. Análise da Explicabilidade (CLFE - LIME/SHAP)

A abordagem CLFE incorpora **LIME e SHAP**, gerados na ronda final, para fornecer interpretabilidade ao modelo de classificação linear treinado sobre o dataset IoT. Isto permite:
- Identificar as features com maior impacto nas classificações (globalmente via SHAP, localmente via LIME).
- Compreender o comportamento do modelo para instâncias específicas.
- Aumentar a transparência e confiança nos resultados da CLFE.
- Visualizar a importância de categorias específicas de features (dl_bitrate, ul_bitrate, features temporais, etc.)

A explicabilidade é apresentada através de gráficos (`lime_final.png`, `shap_final.png`) e relatórios detalhados (`lime_final.html`, `lime_explanation.txt`, `shap_values.npy`), além de visualizações especializadas para features temporais e por categoria.

## 3. Limitações Identificadas (Contexto CLFE - Dataset IoT)

A avaliação da CLFE com o dataset IoT permitiu identificar algumas limitações específicas:
- Convergência extremamente rápida, sugerindo que o problema pode ser relativamente simples para um modelo linear
- Valores perfeitos ou quase perfeitos em algumas métricas (precision, recall), levantando questões sobre possível overfitting
- Necessidade de verificação com dados totalmente novos para confirmar robustez
- Incompatibilidade entre features do modelo e do dataset original (questões com "extra_features" sendo resolvidas)

## 4. Recomendações e Trabalhos Futuros

Com base na análise da CLFE:
- Verificar o desempenho com mais clientes e diferentes distribuições de dados
- Refinar a análise de explicabilidade para entender quais features específicas permitem a alta performance do modelo
- Explorar maneiras de melhorar a interpretação visual dos resultados para não especialistas
- Resolver questões pendentes de incompatibilidade de features para garantir explicabilidade totalmente confiável
- Testar o modelo em ambientes reais para confirmar o desempenho em produção

---

## Checklist de Validação da Secção de Comparação

Utiliza esta checklist antes de integrares a secção de comparação na dissertação:

- [ ] **Tabela comparativa preenchida:** Todos os campos relevantes têm valores reais ou justificação para ausência.
- [ ] **Métricas claras:** Accuracy, precision, recall, F1-score, perdas de treino e validação, tempos de execução e variação de métricas estão presentes para ambas as abordagens.
- [ ] **Análise quantitativa:** O texto reflete e interpreta os valores apresentados, destacando diferenças e possíveis causas.
- [ ] **Secção de explicabilidade:** Explica claramente como LIME e SHAP foram aplicados e o que trouxeram de valor à análise.
- [ ] **Limitações bem identificadas:** Inclui limitações técnicas, de modelo, de dados e de visualização.
- [ ] **Recomendações práticas:** Sugere melhorias concretas e próximos passos para o projeto.
- [ ] **Consistência linguística:** O texto está em português europeu, claro e formal.
- [ ] **Referências cruzadas:** O texto faz referência à tabela e aos ficheiros de relatório/dados, se relevante.
- [ ] **Pronto para integração:** O conteúdo pode ser transposto diretamente para a dissertação sem necessidade de grandes ajustes.

_Esta checklist serve para garantir a qualidade, clareza e completude da análise comparativa._

---

## Execução dos Clientes Federados CLFE (Docker)

- Para levantar múltiplos clientes federados, utilize:
  ```bash
  docker-compose -f docker-compose.generated.yml up --build
  ```
  Cada container será iniciado com o seu `cid` correto e partilhará volumes para reports, results e dataset.

- O script `generate_compose.py` está na raiz da pasta `CLFE/`.
- O volume `DatasetIOT/` é montado como read-only; `reports/` e `results/` são persistentes.
- Para personalizar volumes, nomes de container ou adicionar serviços extra (ex: servidor FL), basta ajustar o template no script.
- O pipeline federado será integrado no cliente após validação da infraestrutura.

## Execução Local de Cliente CLFE

```bash
python client.py --cid=1 --num_clients=4
```

## Notas de Infraestrutura
- Finalizar a configuração Docker (Dockerfile e docker-compose.yml)
- Integrar pipeline de treino federado e explicabilidade no cliente CLFE
- Testar execução distribuída e outputs persistentes
- Atualizar documentação após validação

---

Este documento serve como referência para a redação da dissertação e para a apresentação dos resultados comparativos do projeto.
