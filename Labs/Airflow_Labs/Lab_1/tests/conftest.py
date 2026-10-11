import os
import sys

# Make `src` (inside dags/) importable, exactly as Airflow does for DAG files.
DAGS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dags")
sys.path.insert(0, DAGS_DIR)
