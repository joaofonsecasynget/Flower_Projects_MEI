FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY client/ ./client/
COPY DatasetIOT/ ./DatasetIOT/

# Por omissão, entra na pasta client
WORKDIR /app/client

ENTRYPOINT ["python", "client.py"]
# Os argumentos --cid, --num_clients e --num_total_clients serão passados pelo docker-compose
