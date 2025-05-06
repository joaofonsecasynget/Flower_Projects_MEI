# Estado Atual do Projeto – CLFE (Classificação Linear Federada Explicável)

> Última atualização: 2025-05-06 11:05

---

## 1. Visão Geral

| Item | Descrição |
|------|-----------|
| **Objetivo** | Detectar ataques IoT via Aprendizagem Federada com modelo linear + explicabilidade (LIME/SHAP) |
| **Dataset** | `DatasetIOT/transformed_dataset_imeisv_8642840401612300.csv` |
| **Arquitetura** | Flower FedAvg • Docker Compose • N clientes + 1 servidor |
| **Componentes-chave** | `client/`, `server/`, `explainability/` (metadados, relatórios), `generate_compose.py` |

---

## 2. Kanban Resumido

| Status | Tarefa | Categoria |
|--------|--------|-----------|
| ✅ | Contêiner único (Dockerfile) e compose gerado | Infraestrutura |
| ✅ | Particionamento estratificado & normalização | Dados |
| ✅ | Conversão RLFE → **CLFE** (sigmoid + BCE) | Modelo |
| ✅ | Métricas de classificação (acc/prec/recall/F1) | Modelo |
| ✅ | Sistema de **metadados de features** | Explainability |
| ✅ | Gráficos + relatório HTML (valores originais) | Explainability |
| ✅ | Módulo `export_utils.py` unifica exportação de artefatos | Infraestrutura |
| 🔄 | Saída de explicações em `instance_explanations/` | Explainability |
| 🔄 | Investigar *extra_feature_X* & alinhar datasets | Dados |
| 🔄 | Executar cenários com vários clientes/rondas | Experimentos |
| ☐ | Portar **ADF** para o dataset IoT | Comparação |
| ☐ | Adicionar testes unitários/CI | Qualidade |
| ☐ | Remover código legado `*_fix.py` / `.bak` | Manutenção |

Legenda: ✅ Concluído 🔄 Em progresso ☐ Pendente

---

## 3. Roadmap Detalhado

### 3.1 Infraestrutura
- ✅ Dockerfile único (server + client)
- ✅ Healthcheck no servidor
- ✅ Script `generate_compose.py` para N clientes
- ☐ Parametrizar `NUM_ROUNDS` via CLI/ENV

### 3.2 Pipeline de Dados
- ✅ Remoção de identificadores (`imeisv`, `indice`)
- ✅ Extração de features temporais (`hour`, `dayofweek`, …)
- 🔄 Resolver incompatibilidade **965 vs 961** features → eliminar `extra_feature_X`
- ☐ Validar sistema de metadados com testes automáticos

### 3.3 Modelo & Treino Federado
- ✅ LinearClassificationModel (sigmoid) + BCE
- ✅ Agregação de métricas no servidor
- 🔄 Avaliar overfitting (accuracy ≈ 99 %)
- ☐ Explorar regularização / múltiplas camadas se necessário

### 3.4 Explicabilidade
- ✅ LIME Top-10 & completo, cores verde/vermelho
- ✅ SHAP global + categorias específicas (`dl_bitrate`, …)
- 🔄 Gravar todas as explicações em `instance_explanations/`
- ☐ Incluir categoria **time_features** nos gráficos agregados
- ☐ Métricas de confiabilidade por feature
- ☐ Interface para comparar falsos positivos × falsos negativos

### 3.5 Comparação Formal
- 🔄 Preencher `COMPARACAO_FORMAL.md` com execuções CLFE (2/4/8 clientes, 5/10 rondas)
- ☐ Executar ADF no mesmo dataset e atualizar tabela

### 3.6 Qualidade & Automação
- ☐ Criar testes unitários (pytest) para:
  - `feature_metadata.align_datasets()`
  - `ModelExplainer` categorias/cores
- ☐ Configurar GitHub Actions para lint + testes

### 3.7 Documentação
- ✅ README.md extenso com passos de execução
- 🔄 Atualizar comentários de "RLFE" → "CLFE" no código
- ☐ Tutorial rápido ("Getting Started") no README principal

---

## 4. Próximas Ações Imediatas (Sprint 1 – 1 semana)

1. **Refatorar paths de explicações**   (`client/report_utils.py`)   → `instance_explanations/`  
2. **Experimentos base**   2 clientes × 5 rondas **e** 4 clientes × 5 rondas; salvar métricas JSON  
3. **Extra features bug fix**   usar `feature_metadata.align_datasets` no cliente antes do treino  
4. **Limpeza de código legado**   remover `fix_*` e `.bak` | garantir que não são importados  

---

## 5. Histórico de Atualizações

| Data | Alteração | Autor |
|------|-----------|-------|
| 2025-05-06 | Criação do módulo `export_utils.py` e atualização geral | _AI assistant_ |
| 2025-05-06 | Documento reestruturado: visão geral, kanban, roadmap | _AI assistant_ |
| 2025-05-02 | Conversão RLFE → CLFE, métricas de classificação | JF |
| 2025-05-01 | Infraestrutura Docker/Compose concluída | JF |

---

_Ficheiro serve como "source of truth" para acompanhar o progresso do projeto. Mantenha-o curto, factual e atualize sempre que concluir ou iniciar uma tarefa._