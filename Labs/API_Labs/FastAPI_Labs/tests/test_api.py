"""API tests: run `pytest -v` from the lab folder."""
import joblib
import pytest

SETOSA = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}
VERSICOLOR = {"sepal_length": 6.4, "sepal_width": 3.2, "petal_length": 4.5, "petal_width": 1.5}
VIRGINICA = {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5}


# ---------------- operations ----------------
def test_health_is_always_ok(client_without_model):
    response = client_without_model.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_loaded_model(client):
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {"status": "ready", "model_version": "test-1.0"}


def test_ready_is_503_without_model(client_without_model):
    response = client_without_model.get("/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


# ---------------- v1 prediction ----------------
@pytest.mark.parametrize("flower, species", [(SETOSA, "setosa"), (VERSICOLOR, "versicolor"),
                                             (VIRGINICA, "virginica")])
def test_predict_each_species(client, flower, species):
    response = client.post("/v1/predict", json=flower)
    assert response.status_code == 200
    body = response.json()
    assert body["species"] == species
    assert body["model_version"] == "test-1.0"
    assert set(body["probabilities"]) == {"setosa", "versicolor", "virginica"}
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-6


def test_string_numbers_are_converted(client):
    response = client.post("/v1/predict", json={k: str(v) for k, v in SETOSA.items()})
    assert response.status_code == 200
    assert response.json()["species"] == "setosa"


def test_batch_predicts_every_item_in_order(client):
    response = client.post("/v1/predict/batch", json={"items": [SETOSA, VERSICOLOR, VIRGINICA]})
    assert response.status_code == 200
    body = response.json()
    assert body["count"] == 3
    assert [p["species"] for p in body["predictions"]] == ["setosa", "versicolor", "virginica"]


def test_model_info(client):
    response = client.get("/v1/model")
    assert response.status_code == 200
    body = response.json()
    assert body["model_version"] == "test-1.0"
    assert body["features"] == ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    assert body["classes"] == ["setosa", "versicolor", "virginica"]
    assert 0.8 <= body["test_accuracy"] <= 1.0


# ---------------- validation (all return 422 in one error format) ----------------
@pytest.mark.parametrize("payload, bad_field", [
    ({**SETOSA, "petal_length": -1.0}, "petal_length"),       # negative
    ({**SETOSA, "sepal_width": 0}, "sepal_width"),            # zero
    ({**SETOSA, "petal_width": 50}, "petal_width"),           # unrealistically large
    ({**SETOSA, "sepal_length": "five"}, "sepal_length"),     # not a number
    ({k: v for k, v in SETOSA.items() if k != "petal_width"}, "petal_width"),   # missing
    ({**SETOSA, "colour": "purple"}, "colour"),               # unknown field
])
def test_invalid_input_is_rejected(client, payload, bad_field):
    response = client.post("/v1/predict", json=payload)
    assert response.status_code == 422
    body = response.json()
    assert body["error"]["code"] == "validation_error"
    assert any(bad_field in detail["field"] for detail in body["error"]["details"])
    assert body["request_id"]


def test_empty_batch_is_rejected(client):
    assert client.post("/v1/predict/batch", json={"items": []}).status_code == 422


def test_oversized_batch_is_rejected(client):
    assert client.post("/v1/predict/batch", json={"items": [SETOSA] * 101}).status_code == 422


def test_unknown_route_uses_error_format(client):
    response = client.get("/v1/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


# ---------------- model availability ----------------
def test_predict_is_503_without_model(client_without_model):
    response = client_without_model.post("/v1/predict", json=SETOSA)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "model_unavailable"


def test_model_with_wrong_feature_order_is_refused(monkeypatch, tmp_path, model_path):
    import main
    from fastapi.testclient import TestClient

    bundle = joblib.load(model_path)
    bundle["metadata"] = {**bundle["metadata"], "features": list(reversed(bundle["metadata"]["features"]))}
    bad_path = tmp_path / "wrong_order.pkl"
    joblib.dump(bundle, bad_path)
    monkeypatch.setattr(main, "MODEL_PATH", bad_path)
    with TestClient(main.app) as c:
        response = c.get("/ready")
    assert response.status_code == 503
    assert "feature order mismatch" in response.json()["error"]["message"]


# ---------------- legacy endpoint and API metadata ----------------
def test_legacy_predict_still_works_and_is_flagged_deprecated(client):
    response = client.post("/predict", json=SETOSA)
    assert response.status_code == 200
    assert response.json() == {"response": 0}
    assert response.headers["Deprecation"] == "true"
    assert "/v1/predict" in response.headers["Link"]
    assert client.get("/openapi.json").json()["paths"]["/predict"]["post"]["deprecated"] is True


def test_every_response_has_request_id_and_timing(client):
    response = client.get("/health", headers={"X-Request-ID": "my-trace-123"})
    assert response.headers["X-Request-ID"] == "my-trace-123"
    assert float(response.headers["X-Process-Time-ms"]) >= 0
    assert client.get("/health").headers["X-Request-ID"]   # generated when not sent
