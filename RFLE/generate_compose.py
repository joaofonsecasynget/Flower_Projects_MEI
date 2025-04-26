# Uso: python generate_compose.py <NUM_CLIENTS> <NUM_TOTAL_CLIENTS>
import sys

NUM_CLIENTS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
NUM_TOTAL_CLIENTS = int(sys.argv[2]) if len(sys.argv) > 2 else 10

header = """version: '3.8'

services:
"""

service_template = """
  client_{cid}:
    build: .
    container_name: client_{cid}
    volumes:
      - ./client/reports:/app/client/reports
      - ./client/results:/app/client/results
      - ./DatasetIOT:/app/DatasetIOT:ro
    working_dir: /app/client
    command: ["--cid", "{cid}", "--num_clients", "{num_clients}", "--num_total_clients", "{num_total_clients}"]
"""

with open("docker-compose.generated.yml", "w") as f:
    f.write(header)
    for cid in range(1, NUM_CLIENTS + 1):
        f.write(service_template.format(cid=cid, num_clients=NUM_CLIENTS, num_total_clients=NUM_TOTAL_CLIENTS))
