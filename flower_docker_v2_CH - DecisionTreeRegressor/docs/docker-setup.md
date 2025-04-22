# Configuração Docker

Este documento detalha a configuração Docker do projeto de treino federado com Árvore de Decisão.

## Estrutura

### Containers
1. **Server**
   - Coordena o treino federado
   - Agrega modelos dos clientes
   - Porta: 9091

2. **Client_1 & Client_2**
   - Treinam modelos localmente
   - Geram visualizações e relatórios
   - Compartilham apenas parâmetros agregados

### Volumes
- `/app/reports`: Armazena relatórios e visualizações
- `/app/results`: Armazena modelos treinados
- `/app/dsCaliforniaHousing`: Dataset

## Configuração

### Server
```dockerfile
FROM python:3.12.6-slim

WORKDIR /app
RUN apt-get update && apt-get install -y netcat-traditional
COPY server.py requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 9091
CMD ["python", "server.py"]
```

### Clients
```dockerfile
FROM python:3.12.6-slim

WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libc-dev \
    libssl-dev \
    libpq-dev \
    zlib1g-dev

# Python packages
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Arquivos do projeto
COPY *.py start.sh /app/

# Diretórios
RUN mkdir -p /app/dsCaliforniaHousing /app/reports /app/results && \
    chmod -R 777 /app/reports /app/results

CMD ["/app/start.sh"]
```

### Docker Compose
```yaml
services:
  server:
    build: ./server
    ports:
      - "9091:9091"
    networks:
      - flower-net
    healthcheck:
      test: ["CMD", "nc", "-z", "localhost", "9091"]
      interval: 5s
      timeout: 5s
      retries: 5

  client_1:
    build: ./client_1
    volumes:
      - ./client_1:/app
    networks:
      - flower-net
    depends_on:
      server:
        condition: service_healthy

  client_2:
    build: ./client_2
    volumes:
      - ./client_2:/app
    networks:
      - flower-net
    depends_on:
      server:
        condition: service_healthy

networks:
  flower-net:
    driver: bridge
```

## Dependências

### Server
```txt
flwr==1.5.0
numpy>=1.21.0
```

### Clients
```txt
flwr==1.5.0
numpy>=1.26.0
pandas>=2.2.0
scikit-learn>=1.4.0
matplotlib>=3.8.0
jinja2>=2.11.3
```

## Uso

### Construção
```bash
docker-compose build
```

### Execução
```bash
docker-compose up
```

### Monitoramento
```bash
# Logs
docker-compose logs -f

# Status
docker-compose ps
```

### Limpeza
```bash
docker-compose down
docker system prune
```

## Volumes e Persistência

### Estrutura
```
client_X/
├── reports/
│   ├── tree_structure_roundX.txt
│   ├── feature_importance_roundX.png
│   └── final_report.html
├── results/
│   └── model_client_X.pkl
└── dsCaliforniaHousing/
    └── housing.csv
```

### Permissões
- Reports: 777 (leitura/escrita para todos)
- Results: 777 (leitura/escrita para todos)
- Código: 755 (executável)

## Rede

### Configuração
- Rede bridge isolada
- Comunicação apenas entre containers
- Server exposto na porta 9091

### Healthcheck
- Verifica disponibilidade do servidor
- Intervalo: 5 segundos
- Timeout: 5 segundos
- Máximo de 5 tentativas

## Segurança

### Boas Práticas
1. Imagens slim para menor superfície de ataque
2. Dependências com versões fixas
3. Usuário não-root quando possível
4. Volumes com permissões mínimas necessárias

### Considerações
- Rede isolada para comunicação
- Sem exposição de portas desnecessárias
- Healthcheck para garantir disponibilidade

## Troubleshooting

### Problemas Comuns

1. **Erro de Conexão**
   ```
   Solution: Verificar se porta 9091 está livre
   ```

2. **Permissões de Arquivo**
   ```
   Solution: chmod -R 777 client_X/reports client_X/results
   ```

3. **Memória Insuficiente**
   ```
   Solution: Ajustar profundidade da árvore ou limitar dataset
   ```

### Logs

- Server: `docker-compose logs server`
- Clients: `docker-compose logs client_1 client_2`
- Tempo real: `docker-compose logs -f`

## Performance

### Otimizações
1. Imagens slim para menor footprint
2. Cache de layers do Docker
3. Multi-stage builds quando necessário

### Recursos
- CPU: 1-2 cores por container
- RAM: 512MB-1GB por container
- Disco: ~1GB total

## Manutenção

### Atualizações
1. Atualizar versões no requirements.txt
2. Reconstruir imagens: `docker-compose build`
3. Testar nova configuração

### Backup
1. Volume reports: relatórios e visualizações
2. Volume results: modelos treinados
3. Logs do treinamento

## Referências

1. [Docker Documentation](https://docs.docker.com/)
2. [Docker Compose Documentation](https://docs.docker.com/compose/)
3. [Flower Documentation](https://flower.dev/docs/)
4. [Python Docker Official Images](https://hub.docker.com/_/python) 