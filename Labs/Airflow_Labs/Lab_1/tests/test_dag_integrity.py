"""DAG integrity tests: run inside the Airflow container (see README)."""
import pytest

pytest.importorskip("airflow")

try:  # location changed across Airflow 3.x releases
    from airflow.dag_processing.dagbag import DagBag
except ImportError:
    from airflow.models import DagBag

from conftest import DAGS_DIR


@pytest.fixture(scope="module")
def dagbag():
    return DagBag(dag_folder=DAGS_DIR, include_examples=False)


def test_no_import_errors(dagbag):
    assert dagbag.import_errors == {}, dagbag.import_errors


def test_both_dags_load(dagbag):
    assert {"credit_card_data_ingestion", "kmeans_customer_segmentation"} <= set(dagbag.dag_ids)


def test_training_dag_structure(dagbag):
    dag = dagbag.get_dag("kmeans_customer_segmentation")
    expected = {
        "data_quality.validate", "data_quality.check", "quality_alert",
        "model_training.preprocess", "model_training.candidate_ks",
        "model_training.train_kmeans", "model_training.select_best_k",
        "evaluation.evaluate", "evaluation.publish_report",
    }
    assert expected == set(dag.task_ids)
    # The branch must lead to exactly the two possible paths.
    branch = dag.get_task("data_quality.check")
    assert branch.downstream_task_ids == {"model_training.preprocess", "quality_alert"}
    # Training tasks retry; the alert must not (it fails on purpose).
    assert dag.get_task("model_training.train_kmeans").retries == 2
    assert dag.get_task("quality_alert").retries == 0


def test_ingestion_dag_is_scheduled_and_produces_asset(dagbag):
    dag = dagbag.get_dag("credit_card_data_ingestion")
    task = dag.get_task("land_raw_data")
    assert [o.name for o in task.outlets] == ["credit_card_customers_raw"]
