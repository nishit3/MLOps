"""Flask API that serves the penguin species classifier.

The model and its metadata are read from MODEL_DIR (default /models), which is
either a read-only Docker volume filled by the trainer container (compose) or a
directory baked into the image (standalone `docker build`).

Run in a container with gunicorn (see Dockerfile). For local debugging only:
    MODEL_DIR=artifacts python src/main.py
"""

import json
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/models"))

app = Flask(__name__)


def load_artifacts(model_dir):
    model_path, meta_path = model_dir / "model.keras", model_dir / "metadata.json"
    if not model_path.exists() or not meta_path.exists():
        raise RuntimeError(f"Model artifacts not found in {model_dir}. Did the trainer run successfully?")
    loaded_model = tf.keras.models.load_model(model_path)
    loaded_meta = json.loads(meta_path.read_text())
    return loaded_model, loaded_meta


model, metadata = load_artifacts(MODEL_DIR)
CLASS_NAMES = metadata["class_names"]
NUMERIC = metadata["numeric_features"]
CATEGORICAL = metadata["categorical_features"]  # {feature: [allowed values]}


class InvalidInput(ValueError):
    pass


def parse_features(payload):
    """Validate a request payload and convert it into the model's input dict."""
    features = {}
    for name in NUMERIC:
        raw = payload.get(name)
        if raw in (None, ""):
            raise InvalidInput(f"missing field '{name}'")
        try:
            value = float(raw)
        except (TypeError, ValueError):
            raise InvalidInput(f"field '{name}' must be a number, got {raw!r}")
        if not np.isfinite(value) or value <= 0:
            raise InvalidInput(f"field '{name}' must be a positive number")
        features[name] = tf.constant([[value]], dtype=tf.float32)

    for name, allowed in CATEGORICAL.items():
        value = str(payload.get(name, "")).strip()
        if value not in allowed:
            raise InvalidInput(f"field '{name}' must be one of {allowed}, got {value!r}")
        features[name] = tf.constant([[value]], dtype=tf.string)
    return features


@app.route("/")
def home():
    return render_template("predict.html", metadata=metadata)


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model_version": metadata["model_version"]})


@app.route("/metadata")
def get_metadata():
    return jsonify(metadata)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html", metadata=metadata)

    payload = request.get_json(silent=True) if request.is_json else request.form
    if not payload:
        return jsonify({"error": "send JSON or form data with the penguin's measurements"}), 400
    try:
        features = parse_features(payload)
    except InvalidInput as exc:
        return jsonify({"error": str(exc)}), 400

    probs = model(features, training=False).numpy()[0]
    best = int(np.argmax(probs))
    return jsonify({
        "predicted_class": CLASS_NAMES[best],
        "confidence": round(float(probs[best]), 4),
        "probabilities": {name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)},
        "model_version": metadata["model_version"],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
