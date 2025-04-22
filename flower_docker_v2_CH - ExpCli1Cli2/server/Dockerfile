# Usar Python como base
FROM python:3.12.6-slim

# Definir diretório de trabalho
WORKDIR /app

# Instalar netcat para healthcheck
RUN apt-get update && apt-get install -y netcat-traditional && rm -rf /var/lib/apt/lists/*

# Copiar o ficheiro do servidor
COPY server.py /app/
COPY requirements.txt /app/

# Instalar as dependências do requirements
RUN pip install --no-cache-dir -r requirements.txt

# Expor a porta
EXPOSE 9091

# Comando para iniciar o servidor
CMD ["python", "server.py"]
