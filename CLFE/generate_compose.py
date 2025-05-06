# Uso: python RLFE/generate_compose.py <NUM_CLIENTS> <NUM_TOTAL_CLIENTS>
# Gera um ficheiro docker-compose para a abordagem RLFE (Regressão Linear Federada Explicável)
# Permite simular múltiplos clientes de aprendizagem federada
import sys
import shutil
import glob
import os

NUM_CLIENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
NUM_TOTAL_CLIENTS = int(sys.argv[2]) if len(sys.argv) > 2 else 10

# Limpeza dos outputs antes de gerar o compose
for d in glob.glob("client/reports/client_*"):
    if os.path.isdir(d):
        for f in os.listdir(d):
            fp = os.path.join(d, f)
            if os.path.isfile(fp):
                os.remove(fp)
for d in glob.glob("client/results/client_*"):
    if os.path.isdir(d):
        for f in os.listdir(d):
            fp = os.path.join(d, f)
            if os.path.isfile(fp):
                os.remove(fp)

header = """
services:
  # Servidor Flower para coordenar a aprendizagem federada na abordagem RLFE
  server:
    build:
      context: .
      dockerfile: Dockerfile
    command: ["python", "/app/server.py"]
    ports:
      - "9091:9091"
    networks:
      - flower-net
    healthcheck:
      test: ["CMD", "nc", "-z", "127.0.0.1", "9091"]
      interval: 5s
      timeout: 5s
      retries: 5
"""

service_template = """
  client_{cid}:
    build:
      context: .
      dockerfile: Dockerfile
    container_name: client_{cid}
    volumes:
      - ./client/reports:/app/reports
      - ./client/results:/app/results
      - ./DatasetIOT:/app/DatasetIOT:ro
    working_dir: /app
    command: ["python", "-m", "client.client", "--cid", "{cid}", "--num_clients", "{num_clients}", "--num_total_clients", "{num_total_clients}", "--server_address", "server:9091"]
    networks:
      - flower-net
    depends_on:
      server:
        condition: service_healthy
"""

footer = """
networks:
  flower-net:
    driver: bridge
"""

with open("docker-compose.generated.yml", "w") as f:
    f.write(header)
    for cid in range(1, NUM_CLIENTS + 1):
        f.write(service_template.format(cid=cid, num_clients=NUM_CLIENTS, num_total_clients=NUM_TOTAL_CLIENTS))
    f.write(footer)
