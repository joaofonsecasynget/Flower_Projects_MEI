# Configuração do Docker

Este guia detalha a configuração e uso do Docker no projeto.

## Estrutura dos Containers

### Servidor
```dockerfile
FROM python:3.12.6-slim
WORKDIR /app
EXPOSE 9091
```
- Imagem base Python slim
- Expõe porta 9091 para comunicação
- Configuração mínima para servidor Flower

### Clientes
```dockerfile
FROM python:3.12.6-slim
WORKDIR /app
```
- Mesma imagem base
- Volumes montados para persistência
- Dependências específicas para ML

## Requisitos de Sistema

### Hardware Mínimo
- CPU: 2 cores
- RAM: 4GB
- Disco: 10GB livre

### Software
- Docker 20.10+
- Docker Compose 2.0+
- Portas livres: 9091

## Configuração

### Variáveis de Ambiente
```env
PYTHONUNBUFFERED=1
DOCKER_BUILDKIT=1
```

### Volumes
```yaml
volumes:
  - ./client_1:/app  # Cliente 1
  - ./client_2:/app  # Cliente 2
```

### Redes
```yaml
networks:
  flower-net:
    driver: bridge
```

## Execução

### Construção
```bash
docker-compose build --no-cache
```

### Inicialização
```bash
docker-compose up -d
```

### Monitoramento
```bash
docker-compose logs -f
```

### Parada
```bash
docker-compose down
```

## Troubleshooting

### Problemas Comuns

1. **Erro de Permissão**
```bash
sudo chown -R $USER:$USER .
chmod -R 755 .
```

2. **Porta em Uso**
```bash
sudo lsof -i :9091
kill -9 <PID>
```

3. **Memória Insuficiente**
```bash
docker system prune -a
```

### Logs

#### Servidor
```bash
docker-compose logs server
```

#### Clientes
```bash
docker-compose logs client_1
docker-compose logs client_2
```

## Manutenção

### Limpeza
```bash
docker system prune -a
docker volume prune
```

### Atualização
```bash
docker-compose pull
docker-compose build --no-cache
```

### Backup
```bash
docker cp client_1:/app/results ./backup/
docker cp client_2:/app/results ./backup/
```

## Segurança

### Boas Práticas
- Usar imagens oficiais
- Manter containers atualizados
- Limitar recursos por container
- Usar redes isoladas

### Configurações
```yaml
security_opt:
  - no-new-privileges:true
```

## Otimizações

### Performance
- BuildKit habilitado
- Camadas minimizadas
- Multi-stage builds quando necessário

### Recursos
```yaml
deploy:
  resources:
    limits:
      cpus: '1.0'
      memory: 2G
```

## Referências
- [Docker Docs](https://docs.docker.com/)
- [Docker Compose Docs](https://docs.docker.com/compose/)
- [Python Docker Hub](https://hub.docker.com/_/python) 