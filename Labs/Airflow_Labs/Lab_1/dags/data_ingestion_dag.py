"""DAG 1 of 2: ingest the raw customer data and publish it as an Asset.

Runs on a cron schedule (06:00 UTC daily) or manually. When its task succeeds,
Airflow records an update to the `credit_card_customers_raw` Asset, which
automatically triggers the training DAG (kmeans_training_dag.py).

Trigger it with `inject_bad_data = true` to land corrupted data and watch the
training DAG's data-quality branch stop the pipeline.
"""
from datetime import timedelta

import pendulum
from airflow.sdk import Param, dag, get_current_context, task

from src.config import CUSTOMER_DATA, RAW_SNAPSHOT, SOURCE_DATA


@dag(
    dag_id="credit_card_data_ingestion",
    description="Lands the credit-card customer dataset and publishes it as an Asset",
    schedule="0 6 * * *",
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    tags=["lab1", "ingestion", "asset-producer"],
    default_args={"owner": "Nishit Chaudhary", "retries": 2, "retry_delay": timedelta(seconds=15)},
    params={
        "inject_bad_data": Param(
            False,
            type="boolean",
            description="Corrupt the landed data (nulls + negative values) to test the quality gate.",
        ),
    },
    doc_md=__doc__,
)
def credit_card_data_ingestion():
    @task(outlets=[CUSTOMER_DATA])
    def land_raw_data():
        from src import lab

        context = get_current_context()
        result = lab.snapshot_source_data(
            SOURCE_DATA, RAW_SNAPSHOT, inject_bad_data=context["params"]["inject_bad_data"]
        )
        # Metadata attached to the Asset event; visible in the Assets view of the UI.
        context["outlet_events"][CUSTOMER_DATA].extra = {
            "rows": result["rows"],
            "bad_data_injected": result["bad_data_injected"],
        }
        print(f"Landed {result['rows']} rows at {result['path']} "
              f"(bad data injected: {result['bad_data_injected']})")
        return result

    land_raw_data()


credit_card_data_ingestion()
