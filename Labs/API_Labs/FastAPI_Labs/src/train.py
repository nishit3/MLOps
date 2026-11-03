"""Train the decision tree and save it, together with its metadata, as one versioned bundle.

    python train.py                     # writes ../model/iris_model.pkl, version 1.0.0
    python train.py --version 1.1.0     # same model, new version label

The bundle is a dict {"estimator": <model>, "metadata": {...}}, so the model and the facts the
API reports about it (version, accuracy, feature order, class names) can never get out of sync.
"""
import argparse
import os
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from data import load_data, split_data

DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "model" / "iris_model.pkl"
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]   # order of iris.data columns
HYPERPARAMETERS = {"max_depth": 3, "random_state": 12}                      # unchanged from the original lab


def fit_model(X_train, y_train):
    """Train the same Decision Tree Classifier as the original lab."""
    return DecisionTreeClassifier(**HYPERPARAMETERS).fit(X_train, y_train)


def train_and_save(output_path=DEFAULT_OUTPUT, version="1.0.0"):
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = fit_model(X_train, y_train)

    metadata = {
        "model_version": version,
        "model_type": type(model).__name__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "test_accuracy": round(float(accuracy_score(y_test, model.predict(X_test))), 4),
        "features": FEATURES,
        "classes": [str(name) for name in load_iris().target_names],
        "hyperparameters": HYPERPARAMETERS,
        "sklearn_version": sklearn.__version__,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    joblib.dump({"estimator": model, "metadata": metadata}, tmp_path)
    os.replace(tmp_path, output_path)   # atomic: a running API never sees a half-written file
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", default="1.0.0", help="version label stored with the model")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="where to write the model bundle")
    args = parser.parse_args()
    meta = train_and_save(args.output, args.version)
    print(f"Saved model version {meta['model_version']} to {args.output} "
          f"(test accuracy {meta['test_accuracy']:.4f})")
