# Sistema de Metadados para Features RLFE

Este sistema foi implementado para garantir consistência nas features entre as fases de treinamento e explicabilidade, evitando problemas como:

- **Features extras com nomes genéricos**: Agora todas as features derivadas recebem nomes significativos
- **Categorização inconsistente**: Definição centralizada das categorias das features
- **Valores normalizados vs. originais**: Rastreamento completo com conversão bidirecional
- **Falta de documentação**: Registro claro de todas as transformações aplicadas

## Como Usar o Sistema

### 1. Gerar Metadados a partir do Dataset Original

Execute o script `generate_feature_metadata.py` para criar o arquivo de metadados:

```bash
python RLFE/scripts/generate_feature_metadata.py --dataset RLFE/DatasetIOT/transformed_dataset_imeisv_8642840401612300.csv
```

Isso irá gerar um arquivo `feature_metadata.json` no mesmo diretório que o dataset.

### 2. Consultar Metadados nas Aplicações

```python
from RLFE.explainability.feature_metadata import feature_metadata

# Carregar metadados (geralmente feito automaticamente)
feature_metadata.load()

# Obter a categoria de uma feature
categoria = feature_metadata.get_category('dayofweek')  # Retorna 'time_features'

# Obter todas as features de uma categoria
features_temporais = feature_metadata.get_features_by_category('time_features')

# Converter valores entre original e normalizado
valor_normalizado = feature_metadata.normalize_value('dl_bitrate_0', 157.43)
valor_original = feature_metadata.denormalize_value('dl_bitrate_0', 1.25)
```

## Estrutura dos Metadados

O arquivo `feature_metadata.json` contém:

```json
{
  "features": {
    "dl_bitrate_0": {
      "type": "numeric",
      "category": "dl_bitrate",
      "used_in_training": true,
      "normalization": {
        "mean": 23.45,
        "std": 5.67,
        "min": 0,
        "max": 100
      }
    },
    "_time": {
      "type": "datetime",
      "category": "identifier",
      "used_in_training": false,
      "derived_features": ["hour", "day", "month", "dayofweek", "is_weekend"]
    }
  },
  "categories": {
    "dl_bitrate": ["dl_bitrate_0", "dl_bitrate_1", ...],
    "time_features": ["hour", "day", "month", "dayofweek", "is_weekend"],
    "identifier": ["indice", "imeisv", "_time"]
  }
}
```

## Benefícios do Sistema

1. **Consistência**: Nomes e valores de features padronizados entre treinamento e explicabilidade
2. **Rastreabilidade**: Registro claro da origem de cada feature e suas transformações
3. **Transparência**: Documentação explícita das features usadas no treinamento vs. explicabilidade
4. **Manutenção**: Centralização das definições para fácil atualização

## Integração com Explicabilidade

O sistema já está integrado com:
- `data_loader.py`: Para carregar e preparar dados com features consistentes
- `explainers.py`: Para categorização correta nas visualizações LIME/SHAP

## Recomendações de Uso

1. Gere o arquivo de metadados **antes** de iniciar o treinamento dos clientes
2. Compartilhe o mesmo arquivo entre os clientes para garantir consistência
3. Para novos projetos, defina categorias personalizadas de features antes do treinamento
4. Mantenha o arquivo de metadados versionado junto com o código do projeto
