FROM python:3.10-slim

WORKDIR /app

# Instala netcat-openbsd para healthcheck
RUN apt-get update && apt-get install -y netcat-openbsd && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY client/ ./client/
COPY DatasetIOT/ ./DatasetIOT/
COPY server/server.py ./server.py

# Por omissão, entra na pasta client
WORKDIR /app/client

# ENTRYPOINT removido para permitir comando customizado via docker-compose
