"""Train a TensorFlow/Keras classifier on the Palmer Penguins dataset.

All configuration comes from environment variables so the same image can be
re-run with different settings (see .env / docker-compose.yml):

    DATA_PATH         path to penguins.csv
    OUTPUT_DIR        where model.keras + metadata.json are written
    EPOCHS            max epochs per candidate (early stopping may end sooner)
    MODEL_CANDIDATES  hidden-layer sizes; ';' separates candidates, e.g. "16;32,16"
    DROPOUT           dropout rate after each hidden layer
    MODEL_VERSION     version string recorded in metadata.json
    MIN_ACCURACY      quality gate: exit 1 if test accuracy is below this
    SEED              random seed for reproducibility

Preprocessing (normalization + categorical encoding) lives INSIDE the Keras
model, so the single model.keras artifact accepts raw feature values and the
serving container can never apply different preprocessing than training did.
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

DATA_PATH = os.environ.get("DATA_PATH", "data/penguins.csv")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "artifacts"))
EPOCHS = int(os.environ.get("EPOCHS", "200"))
MODEL_CANDIDATES = os.environ.get("MODEL_CANDIDATES", "16;32,16;64,32,16")
DROPOUT = float(os.environ.get("DROPOUT", "0.1"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "1.0.0")
MIN_ACCURACY = float(os.environ.get("MIN_ACCURACY", "0.90"))
SEED = int(os.environ.get("SEED", "42"))

NUMERIC_FEATURES = ["bill_length_mm", "bill_depth_mm", "flipper_length_mm", "body_mass_g"]
CATEGORICAL_FEATURES = ["island", "sex"]
TARGET = "species"


def load_data(path):
    if not Path(path).exists():
        sys.exit(f"[train] ERROR: dataset not found at '{path}'. See README section 'Get the dataset'.")

    df = pd.read_csv(path)
    required = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
    missing_cols = [c for c in required if c not in df.columns]
    if missing_cols:
        sys.exit(f"[train] ERROR: dataset is missing columns: {missing_cols}")

    rows_before = len(df)
    # Rows without body measurements can't be used; missing sex becomes its own category.
    df = df.dropna(subset=NUMERIC_FEATURES + [TARGET]).copy()
    for col in CATEGORICAL_FEATURES:
        df[col] = df[col].fillna("unknown").astype(str)
    print(f"[train] loaded {rows_before} rows, kept {len(df)} after dropping rows with missing measurements")
    return df


def to_dataset(df, class_names, shuffle=False, batch_size=32):
    features = {c: df[c].to_numpy(dtype="float32").reshape(-1, 1) for c in NUMERIC_FEATURES}
    features.update({c: df[c].to_numpy(dtype=str).reshape(-1, 1) for c in CATEGORICAL_FEATURES})
    labels = df[TARGET].map({name: i for i, name in enumerate(class_names)}).to_numpy(dtype="int32")
    ds = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        ds = ds.shuffle(len(df), seed=SEED)
    return ds.batch(batch_size)


def build_model(train_df, hidden_units, num_classes):
    """Raw features in -> preprocessing layers -> dense layers -> class probabilities."""
    inputs, encoded = {}, []

    for col in NUMERIC_FEATURES:
        inp = tf.keras.Input(shape=(1,), name=col, dtype="float32")
        norm = tf.keras.layers.Normalization(name=f"{col}_norm")
        norm.adapt(train_df[col].to_numpy(dtype="float32").reshape(-1, 1))
        inputs[col] = inp
        encoded.append(norm(inp))

    for col in CATEGORICAL_FEATURES:
        inp = tf.keras.Input(shape=(1,), name=col, dtype="string")
        lookup = tf.keras.layers.StringLookup(name=f"{col}_lookup")
        lookup.adapt(train_df[col].to_numpy(dtype=str))
        one_hot = tf.keras.layers.CategoryEncoding(
            num_tokens=lookup.vocabulary_size(), output_mode="one_hot", name=f"{col}_onehot"
        )
        inputs[col] = inp
        encoded.append(one_hot(lookup(inp)))

    x = tf.keras.layers.concatenate(encoded)
    for i, units in enumerate(hidden_units):
        x = tf.keras.layers.Dense(units, activation="relu", name=f"dense_{i}")(x)
        if DROPOUT > 0:
            x = tf.keras.layers.Dropout(DROPOUT, name=f"dropout_{i}")(x)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="species")(x)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def parse_candidates(spec):
    candidates = []
    for chunk in spec.split(";"):
        chunk = chunk.strip()
        if chunk:
            candidates.append([int(u) for u in chunk.split(",")])
    if not candidates:
        sys.exit("[train] ERROR: MODEL_CANDIDATES is empty")
    return candidates


def main():
    tf.keras.utils.set_random_seed(SEED)
    df = load_data(DATA_PATH)
    class_names = sorted(df[TARGET].unique().tolist())

    # 70 / 15 / 15 stratified split
    train_df, temp_df = train_test_split(df, test_size=0.30, stratify=df[TARGET], random_state=SEED)
    val_df, test_df = train_test_split(temp_df, test_size=0.50, stratify=temp_df[TARGET], random_state=SEED)
    print(f"[train] split: train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    train_ds = to_dataset(train_df, class_names, shuffle=True)
    val_ds = to_dataset(val_df, class_names)
    test_ds = to_dataset(test_df, class_names)

    results, best = [], None
    for hidden_units in parse_candidates(MODEL_CANDIDATES):
        name = "-".join(map(str, hidden_units))
        print(f"[train] training candidate {name} (max {EPOCHS} epochs)")
        model = build_model(train_df, hidden_units, len(class_names))
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True
        )
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early_stop], verbose=0)
        val_loss, val_acc = model.evaluate(val_ds, verbose=0)
        result = {
            "hidden_units": hidden_units,
            "epochs_run": len(history.history["loss"]),
            "val_accuracy": round(float(val_acc), 4),
            "val_loss": round(float(val_loss), 4),
        }
        results.append(result)
        print(f"[train]   -> val_accuracy={val_acc:.4f} val_loss={val_loss:.4f} epochs={result['epochs_run']}")

        # Best = highest val accuracy, ties broken by lower val loss
        if best is None or (val_acc, -val_loss) > (best[1]["val_accuracy"], -best[1]["val_loss"]):
            best = (model, result)

    model, best_result = best
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    print(f"[train] selected {best_result['hidden_units']} | test_accuracy={test_acc:.4f}")

    if test_acc < MIN_ACCURACY:
        sys.exit(f"[train] QUALITY GATE FAILED: test accuracy {test_acc:.4f} < MIN_ACCURACY {MIN_ACCURACY}. "
                 "No artifacts written; serving will not start.")

    metadata = {
        "model_version": MODEL_VERSION,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"name": "Palmer Penguins", "rows_used": len(df)},
        "class_names": class_names,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": {c: sorted(df[c].unique().tolist()) for c in CATEGORICAL_FEATURES},
        "selected_model": best_result,
        "candidates": results,
        "test_accuracy": round(float(test_acc), 4),
        "config": {"epochs": EPOCHS, "dropout": DROPOUT, "seed": SEED, "min_accuracy": MIN_ACCURACY},
    }

    # Write to temp names, then rename, so serving never sees a half-written artifact.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_model, tmp_meta = OUTPUT_DIR / "model.tmp.keras", OUTPUT_DIR / "metadata.tmp.json"
    model.save(tmp_model)
    tmp_meta.write_text(json.dumps(metadata, indent=2))
    os.replace(tmp_model, OUTPUT_DIR / "model.keras")
    os.replace(tmp_meta, OUTPUT_DIR / "metadata.json")
    print(f"[train] wrote model.keras and metadata.json to {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()
