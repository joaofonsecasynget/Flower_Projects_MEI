# Comparação Formal das Abordagens Federadas

Este documento apresenta um template de tabela comparativa e uma estrutura detalhada para a secção de comparação formal entre as duas abordagens principais desenvolvidas no projeto:

- **Árvore de Decisão Federada (DecisionTreeRegressor, scikit-learn)**
- **Regressão Linear Federada com Explicabilidade via LIME e SHAP (PyTorch)**

> **Abreviaturas utilizadas:**
> - **ADF**: Árvore de Decisão Federada (DecisionTreeRegressor, scikit-learn)
> - **RLFE**: Regressão Linear Federada Explicável (PyTorch, LIME/SHAP)

---

## 1. Tabela Comparativa de Resultados

| Critério / Métrica                                      | ADF | RLFE |
|---------------------------------------------------------|:---:|:----:|
| **RMSE (Cliente 1)**                                   | 235231.05 | [valor - preencher após teste com novo dataset] |
| **RMSE (Cliente 2)**                                   | 64731.28 | [valor - preencher após teste com novo dataset] |
| **Perda de Validação Média**                           | 54 817 417 452.31 (C1) / 3 851 295 180.66 (C2) | [valor - preencher após teste com novo dataset] |
| **Tempo de Execução (min)**                            | [valor a recolher] | [valor a recolher] |
| **Estabilidade entre rondas (variação de RMSE)**       | [valor a recolher] | [valor a recolher] |
| **Qualidade da Explicabilidade**                       | Explicação nativa (estrutura da árvore, importância das features) | LIME/SHAP, gráficos por ronda e evolução global |
| **Visualização da Explicabilidade**                    | Gráfico da árvore, importância das features | Gráficos LIME/SHAP por ronda e evolução |
| **Interpretação Global**                               | Boa (estrutura da árvore) | Limitada (coeficientes lineares, LIME/SHAP para instâncias) |
| **Interpretação Local**                                | Média (caminho na árvore) | Boa (LIME/SHAP para instâncias) |
| **Robustez à distribuição dos dados**                  | [observação] | [observação] |
| **Limitações Observadas**                              | Problemas de gráficos no Cliente 2; variação entre clientes | RMSE elevado; convergência lenta; valores ausentes (-1) |
| **Explicabilidade (LIME/SHAP)**                        | Explicabilidade intrínseca (estrutura da árvore); sem LIME/SHAP | LIME e SHAP aplicados; análise detalhada das features |

> **Notas:**
> - Preencher os campos [valor] e [observação] com os resultados dos relatórios.
> - Adicionar/remover critérios conforme o foco da análise.
> - O novo dataset (ds_testes_iniciais.csv) está pronto para testes iniciais, desde que se realize pré-processamento: substituição de -1 por NaN, remoção de colunas não relevantes, confirmação do target e normalização das features.
> - Recomenda-se continuar a preencher os campos em falta na tabela à medida que os testes com o novo dataset forem realizados.

---

## 2. Estrutura Sugerida para a Secção de Comparação Formal

### 4. Comparação Formal das Abordagens

#### 4.1. Critérios de Comparação
- Desempenho preditivo (RMSE, perda de validação)
- Estabilidade entre rondas e entre clientes
- Qualidade e utilidade da explicabilidade (global e local)
- Tempo de execução e escalabilidade
- Robustez à distribuição dos dados
- Limitações observadas

#### 4.2. Resultados Quantitativos
- Apresentação da tabela comparativa de métricas (ver acima)
- Discussão dos valores obtidos para cada critério

#### 4.3. Análise da Explicabilidade
- Exemplos de explicações geradas por cada abordagem
- Discussão sobre a utilidade prática das explicações (para utilizadores finais, decisores, etc.)
- Visualizações: gráficos de importância, árvores, LIME/SHAP

#### 4.4. Limitações e Considerações Práticas
- Pontos fortes e fracos de cada abordagem no contexto federado
- Impacto do modelo e da explicabilidade na adoção prática

#### 4.5. Síntese e Recomendações
- Resumo dos principais achados
- Recomendações para uso futuro e para trabalhos subsequentes

---

Este documento serve como referência para a redação da dissertação e para a apresentação dos resultados comparativos do projeto.

## 1. Resultados Quantitativos

A comparação entre as duas abordagens de aprendizagem federada — a Árvore de Decisão Federada (DecisionTreeRegressor, scikit-learn) e a Regressão Linear Federada com Explicabilidade via LIME e SHAP (PyTorch) — foi realizada com base em métricas extraídas dos relatórios finais de cada cliente. As principais métricas analisadas foram o RMSE (Root Mean Squared Error), a perda de treino e a perda de validação.

**Árvore de Decisão Federada**
- Cliente 1:
  - RMSE: 235231.05
  - Perda de treino: 55 658 836 565.70
  - Perda de validação: 54 817 417 452.31
- Cliente 2:
  - RMSE: 64 731.28
  - Perda de treino: 1 799 251 402.08
  - Perda de validação: 3 851 295 180.66

Estes valores evidenciam uma variação significativa entre clientes, possivelmente relacionada com a distribuição dos dados ou características específicas de cada subconjunto.

**Regressão Linear Federada**
- RMSE global: ~235 000

O valor elevado do RMSE na Regressão Linear indica menor capacidade preditiva neste contexto, apesar da sua simplicidade e interpretabilidade.

## 2. Análise da Explicabilidade

A abordagem de Regressão Linear Federada foi enriquecida com técnicas de explicabilidade, nomeadamente LIME e SHAP, permitindo uma análise detalhada do impacto de cada variável na previsão dos preços imobiliários. Estas ferramentas facilitaram a identificação das características mais relevantes para o modelo, promovendo transparência e confiança nos resultados.

Por outro lado, a Árvore de Decisão Federada, apesar de fornecer uma certa interpretabilidade intrínseca (através da análise da estrutura da árvore), não foi complementada com métodos avançados de explicabilidade neste trabalho, limitando a profundidade da análise interpretativa.

## 3. Limitações Identificadas

Durante a execução dos experimentos, foram identificadas várias limitações:
- O elevado valor de RMSE na Regressão Linear sugere que o modelo pode não ser adequado para a complexidade do problema ou que existe necessidade de maior pré-processamento dos dados.
- Foram registados problemas na geração de gráficos no Cliente 2 da Árvore de Decisão, o que dificultou a análise visual dos resultados.
- A discrepância entre os resultados dos clientes indica possível heterogeneidade nos dados, que poderá afetar a generalização dos modelos.
- A utilização exclusiva do dataset California Housing pode limitar a validade externa das conclusões.

## 4. Recomendações e Trabalhos Futuros

Com base nos resultados obtidos, recomenda-se:
- A exploração de modelos mais robustos e adequados à natureza dos dados, como ensembles ou redes neuronais profundas, especialmente para problemas com elevada variabilidade.
- A integração de técnicas de explicabilidade também na abordagem baseada em árvores de decisão, para uma comparação mais equitativa.
- A realização de experiências com diferentes datasets, de modo a validar a generalização dos resultados.
- A melhoria dos mecanismos de visualização e geração de relatórios, garantindo uma análise mais completa e acessível dos resultados.

---

## Checklist de Validação da Secção de Comparação

Utiliza esta checklist antes de integrares a secção de comparação na dissertação:

- [ ] **Tabela comparativa preenchida:** Todos os campos relevantes têm valores reais ou justificação para ausência.
- [ ] **Métricas claras:** RMSE, perdas de treino e validação, tempos de execução e variação de RMSE estão presentes para ambas as abordagens.
- [ ] **Análise quantitativa:** O texto reflete e interpreta os valores apresentados, destacando diferenças e possíveis causas.
- [ ] **Secção de explicabilidade:** Explica claramente como LIME e SHAP foram aplicados e o que trouxeram de valor à análise.
- [ ] **Limitações bem identificadas:** Inclui limitações técnicas, de modelo, de dados e de visualização.
- [ ] **Recomendações práticas:** Sugere melhorias concretas e próximos passos para o projeto.
- [ ] **Consistência linguística:** O texto está em português europeu, claro e formal.
- [ ] **Referências cruzadas:** O texto faz referência à tabela e aos ficheiros de relatório/dados, se relevante.
- [ ] **Pronto para integração:** O conteúdo pode ser transposto diretamente para a dissertação sem necessidade de grandes ajustes.

_Esta checklist serve para garantir a qualidade, clareza e completude da análise comparativa._
