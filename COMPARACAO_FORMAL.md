# Comparação Formal das Abordagens Federadas

Este documento apresenta um template de tabela comparativa e uma estrutura detalhada para a secção de comparação formal entre as abordagens de aprendizagem federada desenvolvidas no projeto, com foco na implementação **RLFE** utilizando o **dataset IoT**:

- **RLFE**: Regressão Linear Federada com Explicabilidade via LIME e SHAP (PyTorch) utilizando o dataset IoT.
- **ADF**: Árvore de Decisão Federada (DecisionTreeRegressor, scikit-learn) - *Nota: Resultados anteriores eram do dataset California Housing. A comparação direta requer testes com ADF no dataset IoT.*

> **Abreviaturas utilizadas:**
> - **ADF**: Árvore de Decisão Federada (DecisionTreeRegressor, scikit-learn)
> - **RLFE**: Regressão Linear Federada Explicável (PyTorch, LIME/SHAP, Dataset IoT)

---

## 1. Tabela Comparativa de Resultados (Foco RLFE no Dataset IoT)

| Critério / Métrica                                      | RLFE (Dataset IoT) | ADF (Dataset California Housing - Referência Antiga) |
|---------------------------------------------------------|:------------------:|:----------------------------------------------------:|
| **RMSE Médio (entre clientes)**                          | [Preencher após execução RLFE com dataset IoT] | ~150k (Média dos valores antigos C1: 235k, C2: 65k) |
| **Perda de Validação Média**                           | [Preencher após execução RLFE com dataset IoT] | ~29B (Média dos valores antigos C1: 55B, C2: 4B)   |
| **Tempo de Execução Médio (por ronda)**                 | [Preencher - medir `fit_duration` + `evaluate_duration` do JSON] | [Não medido anteriormente] |
| **Estabilidade entre rondas (variação de RMSE)**       | [Analisar evolução do RMSE no JSON/plots] | [Não analisado anteriormente] |
| **Qualidade da Explicabilidade (Global)**              | Limitada (Coeficientes lineares) | Boa (Estrutura da árvore) |
| **Qualidade da Explicabilidade (Local)**               | Boa (LIME/SHAP para instâncias específicas) | Média (Caminho na árvore) |
| **Visualização da Explicabilidade**                    | Gráficos LIME (Top10/Completo), SHAP | Gráfico da árvore, importância das features |
| **Robustez à distribuição dos dados**                  | [Observar variação entre clientes nos testes RLFE] | Variação significativa observada entre clientes |
| **Limitações Observadas**                              | [Observar nos testes RLFE - e.g., convergência, escala RMSE] | Problemas de gráficos (antigo); variação entre clientes |

> **Notas:**
> - Preencher os campos [Preencher] e [Analisar/Observar] com os resultados das execuções da RLFE no dataset IoT (`reports/client_X/metrics_history.json`, `reports/client_X/final_report.html`).
> - A coluna ADF serve apenas como referência da implementação anterior com outro dataset. Uma comparação formal exigiria re-executar ADF no dataset IoT.
> - O foco principal é caracterizar o desempenho e a explicabilidade da RLFE no cenário atual.

---

## 2. Estrutura Sugerida para a Secção de Análise (Foco RLFE)

### 4. Análise da Abordagem RLFE com Dataset IoT

#### 4.1. Desempenho Preditivo
- Análise do RMSE e Perda de Validação (média e por cliente).
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
- Discussão sobre a adequação do modelo linear ao dataset IoT (com base no RMSE).
- Observações sobre a heterogeneidade dos dados entre clientes (se aplicável).
- Outras limitações encontradas durante a execução e análise.

#### 4.5. Síntese
- Resumo dos pontos fortes e fracos da abordagem RLFE neste cenário.

---

## 1. Resultados Quantitativos (Contextualização)

A análise quantitativa foca-se agora na abordagem **RLFE (Regressão Linear Federada com Explicabilidade)** implementada com **PyTorch** e utilizando o **dataset IoT**. Resultados anteriores da abordagem ADF (Árvore de Decisão Federada com scikit-learn) pertencem a um contexto diferente (dataset California Housing) e servem apenas como referência histórica.

**Regressão Linear Federada (RLFE) - Dataset IoT**
- **RMSE médio (clientes):** [Preencher após execução]
- **Perda de Validação média (clientes):** [Preencher após execução]
- **Tempos de execução (fit, evaluate):** Disponíveis em `metrics_history.json` por ronda.

A análise detalhada destes valores, incluindo a sua evolução e variação entre clientes, permitirá caracterizar o desempenho da RLFE no problema atual.

## 2. Análise da Explicabilidade (RLFE - LIME/SHAP)

A abordagem RLFE incorpora **LIME e SHAP**, gerados na ronda final, para fornecer interpretabilidade ao modelo de regressão linear treinado sobre o dataset IoT. Isto permite:
- Identificar as features com maior impacto nas previsões (globalmente via SHAP, localmente via LIME).
- Compreender o comportamento do modelo para instâncias específicas.
- Aumentar a transparência e confiança nos resultados da RLFE.

A explicabilidade é apresentada através de gráficos (`lime_final.png`, `shap_final.png`) e relatórios detalhados (`lime_final.html`, `lime_explanation.txt`, `shap_values.npy`).

## 3. Limitações Identificadas (Contexto RLFE - Dataset IoT)

A avaliação da RLFE com o dataset IoT permitirá identificar limitações específicas, tais como:
- Potencial inadequação do modelo linear à complexidade dos dados (refletida no RMSE).
- Desafios relacionados com a escala ou distribuição das features no dataset IoT.
- Variação de desempenho entre clientes devido à heterogeneidade dos dados federados.
- Tempo de execução, especialmente da fase de explicabilidade (embora mitigado por ocorrer só no fim).

## 4. Recomendações e Trabalhos Futuros

Com base na análise da RLFE:
- Avaliar a necessidade de modelos mais complexos (e.g., redes neuronais) se o desempenho da regressão linear for insuficiente.
- Investigar técnicas de pré-processamento específicas para o dataset IoT.
- Aprofundar a análise das explicações LIME/SHAP para extrair insights sobre o domínio do problema.
- Comparar formalmente com outras abordagens *após* estas serem adaptadas e testadas com o mesmo dataset IoT.

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

---

## Execução dos Clientes Federados RLFE (Docker)

- Para levantar múltiplos clientes federados, utilize:
  ```bash
  docker-compose -f docker-compose.generated.yml up --build
  ```
  Cada container será iniciado com o seu `cid` correto e partilhará volumes para reports, results e dataset.

- O script `generate_compose.py` está na raiz da pasta `RLFE/`.
- O volume `DatasetIOT/` é montado como read-only; `reports/` e `results/` são persistentes.
- Para personalizar volumes, nomes de container ou adicionar serviços extra (ex: servidor FL), basta ajustar o template no script.
- O pipeline federado será integrado no cliente após validação da infraestrutura.

## Execução Local de Cliente RLFE

```bash
python client.py --cid=1 --num_clients=4
```

## Notas de Infraestrutura
- Finalizar a configuração Docker (Dockerfile e docker-compose.yml)
- Integrar pipeline de treino federado e explicabilidade no cliente RLFE
- Testar execução distribuída e outputs persistentes
- Atualizar documentação após validação

---

Este documento serve como referência para a redação da dissertação e para a apresentação dos resultados comparativos do projeto.
