"""Loading the trained model bundle once, and making predictions with it.

The original reloaded the model from disk inside every request, using a path that only worked
when the server was started from src/. Here the model is loaded once (at API startup), validated,
and passed to the endpoints.
"""
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

REQUIRED_METADATA = {"model_version", "model_type", "trained_at", "test_accuracy", "features", "classes",
                     "hyperparameters"}


class ModelLoadError(RuntimeError):
    """The model file is missing, in the wrong format, or doesn't match the API's input contract."""


@dataclass(frozen=True)
class LoadedModel:
    estimator: object
    metadata: dict

    @property
    def version(self):
        return self.metadata["model_version"]


def load_model(path, expected_features):
    """Load and validate a model bundle written by train.py."""
    path = Path(path)
    if not path.exists():
        raise ModelLoadError(f"model file not found at {path}; run `python train.py` first")

    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {"estimator", "metadata"} <= bundle.keys():
        raise ModelLoadError("model file is in the old format (a bare estimator); retrain with `python train.py`")

    metadata = bundle["metadata"]
    missing = REQUIRED_METADATA - metadata.keys()
    if missing:
        raise ModelLoadError(f"model metadata is missing {sorted(missing)}")
    if list(metadata["features"]) != list(expected_features):
        raise ModelLoadError(f"feature order mismatch: model expects {metadata['features']}, "
                             f"API sends {list(expected_features)}")
    return LoadedModel(estimator=bundle["estimator"], metadata=metadata)


def predict(model, rows):
    """Predict species and per-class probabilities for a list of feature rows."""
    probabilities = model.estimator.predict_proba(np.asarray(rows, dtype=float))
    class_ids = [int(c) for c in model.estimator.classes_]
    names = model.metadata["classes"]
    results = []
    for row in probabilities:
        best = int(np.argmax(row))
        results.append({
            "class_id": class_ids[best],
            "species": names[class_ids[best]],
            "probabilities": {names[c]: round(float(p), 4) for c, p in zip(class_ids, row)},
        })
    return results
