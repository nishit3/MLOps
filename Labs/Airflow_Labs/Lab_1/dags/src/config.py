"""Shared configuration for both DAGs: paths, the data Asset, and defaults."""
import os
import re

from airflow.sdk import Asset

# dags/ is mounted at /opt/airflow/dags, so the lab root is /opt/airflow inside the containers.
DAGS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_DATA = os.path.join(DAGS_DIR, "data", "file.csv")
TEST_DATA = os.path.join(DAGS_DIR, "data", "test.csv")

# Shared, persistent output folder (bind-mounted to ./working_data on the host).
WORKING_DIR = os.environ.get(
    "PIPELINE_WORKING_DIR", os.path.join(os.path.dirname(DAGS_DIR), "working_data")
)
RAW_SNAPSHOT = os.path.join(WORKING_DIR, "raw", "credit_card_customers.csv")

# The Asset links the two DAGs: ingestion produces it, training is scheduled on it.
CUSTOMER_DATA = Asset(name="credit_card_customers_raw", uri=f"file://{RAW_SNAPSHOT}")

DEFAULT_FEATURES = ["BALANCE", "PURCHASES", "CREDIT_LIMIT"]


def run_dir(run_id: str) -> str:
    """Per-run artifact folder. Run ids contain ':' and '+', which Windows can't store."""
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    path = os.path.join(WORKING_DIR, "runs", safe)
    os.makedirs(path, exist_ok=True)
    return path
