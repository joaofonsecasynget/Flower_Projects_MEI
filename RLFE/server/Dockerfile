FROM python:3.10-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar netcat para healthcheck
RUN apt-get update && apt-get install -y netcat-traditional && rm -rf /var/lib/apt/lists/*

# Copiar o arquivo de requisitos e instalar
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o arquivo do servidor
COPY server.py /app/

# Expor a porta para comunicação com os clientes
EXPOSE 9091

# Comando para iniciar o servidor
CMD ["python", "server.py"]
