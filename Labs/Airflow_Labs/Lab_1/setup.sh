#!/bin/bash
# One-step setup for Linux/macOS (Windows: follow the README steps in PowerShell).
set -e

mkdir -p ./logs ./plugins ./working_data

# On Linux, run Airflow as your own user so files in logs/ and working_data/ are yours.
if [ "$(uname)" = "Linux" ]; then
  sed -i "s/^AIRFLOW_UID=.*/AIRFLOW_UID=$(id -u)/" .env
fi

docker compose down -v --remove-orphans
docker compose build
docker compose up airflow-init
docker compose up -d
echo "Airflow is starting: open http://localhost:8080 (login: airflow / airflow) in about a minute."
