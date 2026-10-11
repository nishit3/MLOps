"""DAG 2 of 2: validate data, train K-Means for many k in parallel, pick the elbow.

Scheduled on the `credit_card_customers_raw` Asset: it runs automatically
whenever the ingestion DAG lands new data. It can also be triggered manually,
with the k range, features and quality thresholds set in the trigger form.

Flow:
    data_quality.validate -> data_quality.check  --(fail)--> quality_alert (fails the run)
                                                 --(pass)--> model_training.preprocess
    model_training.preprocess -> candidate_ks -> train_kmeans[k] (one mapped task per k, parallel)
                              -> select_best_k -> evaluation.evaluate -> evaluation.publish_report
"""
import logging
import os
from datetime import timedelta

import pendulum
from airflow.exceptions import AirflowFailException
from airflow.sdk import Param, TaskGroup, dag, get_current_context, task

from src.config import CUSTOMER_DATA, DEFAULT_FEATURES, RAW_SNAPSHOT, TEST_DATA, WORKING_DIR, run_dir

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Callbacks: run inside the task process, so their output appears in the task log.
# ---------------------------------------------------------------------------
def log_retry(context):
    ti = context["ti"]
    log.warning("RETRYING %s (map_index=%s) after attempt %s: %s",
                ti.task_id, ti.map_index, ti.try_number, context.get("exception"))


def log_failure(context):
    ti = context["ti"]
    log.error("FAILED %s (map_index=%s) in run %s: %s",
              ti.task_id, ti.map_index, context["run_id"], context.get("exception"))


default_args = {
    "owner": "Nishit Chaudhary",
    "retries": 2,
    "retry_delay": timedelta(seconds=5),
    "retry_exponential_backoff": True,      # 5s, then ~10s, ...
    "max_retry_delay": timedelta(minutes=1),
    "execution_timeout": timedelta(minutes=10),
    "on_retry_callback": log_retry,
    "on_failure_callback": log_failure,
}


@dag(
    dag_id="kmeans_customer_segmentation",
    description="Asset-triggered K-Means pipeline with a data-quality gate and parallel training",
    schedule=[CUSTOMER_DATA],
    start_date=pendulum.datetime(2026, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["lab1", "training", "asset-consumer", "dynamic-task-mapping"],
    params={
        "k_min": Param(1, type="integer", minimum=1, maximum=49, description="Smallest k to try"),
        "k_max": Param(20, type="integer", minimum=2, maximum=50, description="Largest k to try"),
        "features": Param(DEFAULT_FEATURES, type="array", items={"type": "string"},
                          description="Columns used for clustering"),
        "min_rows": Param(500, type="integer", minimum=1, description="Quality gate: minimum rows"),
        "max_null_fraction": Param(0.05, type="number", minimum=0, maximum=1,
                                   description="Quality gate: max fraction of nulls per feature"),
        "simulate_failure": Param(False, type="boolean",
                                  description="Make one training task fail on its first attempt to demo retries"),
    },
    doc_md=__doc__,
)
def kmeans_customer_segmentation():

    # ----------------------------- data quality -----------------------------
    with TaskGroup("data_quality", tooltip="Validate the landed data before training"):

        @task
        def validate():
            from src import lab

            p = get_current_context()["params"]
            report = lab.validate_data(RAW_SNAPSHOT, p["features"], p["min_rows"], p["max_null_fraction"])
            log.info("Data-quality report: %s", report)
            return report

        @task.branch
        def check(report: dict):
            # Return the task_id(s) to follow; every other direct downstream task is skipped.
            return "model_training.preprocess" if report["passed"] else "quality_alert"

        report = validate()
        decision = check(report)

    @task(retries=0)
    def quality_alert(report: dict):
        """Fail loudly (red run in the UI) and record why training was skipped."""
        import json

        out = run_dir(get_current_context()["run_id"])
        with open(os.path.join(out, "quality_failure.json"), "w") as f:
            json.dump(report, f, indent=2)
        raise AirflowFailException("Data-quality gate failed, training skipped: " + "; ".join(report["failures"]))

    # ----------------------------- training ---------------------------------
    with TaskGroup("model_training", tooltip="Parallel K-Means training and elbow selection"):

        @task
        def preprocess():
            from src import lab

            ctx = get_current_context()
            return lab.preprocess(RAW_SNAPSHOT, ctx["params"]["features"], run_dir(ctx["run_id"]))

        @task
        def candidate_ks(prep: dict):
            from src import lab

            p = get_current_context()["params"]
            try:
                ks = lab.candidate_ks(p["k_min"], p["k_max"], prep["rows_used"])
            except ValueError as exc:
                raise AirflowFailException(str(exc))  # bad config: retrying won't help
            log.info("Training %d models in parallel: k=%s", len(ks), ks)
            return ks

        # Dynamic task mapping: one task instance per k, created at runtime.
        @task(max_active_tis_per_dagrun=8)
        def train_kmeans(k: int, prep: dict):
            from src import lab

            ctx = get_current_context()
            p, ti = ctx["params"], ctx["ti"]
            if p["simulate_failure"] and k == p["k_min"] + 1 and ti.try_number == 1:
                raise RuntimeError(f"Simulated transient failure for k={k} (succeeds on retry)")
            result = lab.train_kmeans(k, prep["scaled_path"], run_dir(ctx["run_id"]))
            log.info("k=%d SSE=%.4f", k, result["sse"])
            return result

        @task
        def select_best_k(results):
            from src import lab

            selection = lab.select_best_k(list(results), run_dir(get_current_context()["run_id"]))
            log.info("Elbow at k=%d (%s)", selection["best_k"], selection["elbow_method"])
            return selection

        prep = preprocess()
        ks = candidate_ks(prep)
        trained = train_kmeans.partial(prep=prep).expand(k=ks)
        selection = select_best_k(trained)

    # ----------------------------- evaluation -------------------------------
    with TaskGroup("evaluation", tooltip="Score the chosen model and publish a report"):

        @task
        def evaluate(selection: dict, prep: dict):
            from src import lab

            metrics = lab.evaluate(selection["best_model_path"], prep["scaled_path"],
                                   prep["scaler_path"], TEST_DATA, prep["features"])
            log.info("Evaluation: %s", metrics)
            return metrics

        @task
        def publish_report(prep: dict, selection: dict, metrics: dict):
            from src import lab

            ctx = get_current_context()
            summary = {
                "run_id": ctx["run_id"],
                "triggered_by": getattr(ctx["dag_run"].run_type, "value", str(ctx["dag_run"].run_type)),
                "params": dict(ctx["params"]),
                "data": {"rows_in": prep["rows_in"], "rows_used": prep["rows_used"],
                         "features": prep["features"]},
                "best_k": selection["best_k"],
                "elbow_method": selection["elbow_method"],
                "sse_by_k": selection["sse_by_k"],
                **metrics,
            }
            path = lab.write_report(run_dir(ctx["run_id"]), os.path.join(WORKING_DIR, "latest"), summary)
            log.info("Report written to %s", path)
            return summary

        metrics = evaluate(selection, prep)
        publish_report(prep, selection, metrics)

    decision >> [prep, quality_alert(report)]


kmeans_customer_segmentation()
