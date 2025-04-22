FROM python:3.12.6-slim

WORKDIR /app

# Instalar dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libc-dev \
    libssl-dev \
    libpq-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Atualizar pip
RUN pip install --upgrade pip

# Instalar numpy e pandas separadamente para melhor cache
RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir pandas

# Copiar requirements.txt e instalar dependências Python
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copiar arquivos do projeto
COPY client_1.py /app/
COPY utils.py /app/
COPY explainability.py /app/
COPY start.sh /app/

# Criar diretórios necessários
RUN mkdir -p /app/dsCaliforniaHousing && \
    mkdir -p /app/reports && \
    mkdir -p /app/results

# Copiar dados
COPY dsCaliforniaHousing/housing.csv /app/dsCaliforniaHousing/housing.csv

# Configurar permissões
RUN chmod +x /app/start.sh && \
    chown -R root:root /app && \
    chmod -R 755 /app && \
    chmod -R 777 /app/reports && \
    chmod -R 777 /app/results

# Comando para iniciar o cliente
CMD ["/app/start.sh"]
