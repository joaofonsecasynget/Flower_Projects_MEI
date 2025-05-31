FROM python:3.10-slim

WORKDIR /app

# Instala netcat-openbsd para healthcheck e dependências matplotlib
RUN apt-get update && apt-get install -y netcat-openbsd libfreetype6-dev libpng-dev --no-install-recommends && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY client/ ./client/
COPY DatasetIOT/ ./DatasetIOT/
COPY server/server.py ./server.py
COPY explainability/ ./explainability/

# Por omissão, entra na pasta client
WORKDIR /app/client

# ENTRYPOINT removido para permitir comando customizado via docker-compose
