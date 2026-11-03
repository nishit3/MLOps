import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

import main  # noqa: E402
from train import train_and_save  # noqa: E402


@pytest.fixture(scope="session")
def model_path(tmp_path_factory):
    """Train a fresh model for the tests, so they never depend on a committed file."""
    path = tmp_path_factory.mktemp("model") / "iris_model.pkl"
    train_and_save(path, version="test-1.0")
    return path


@pytest.fixture
def client(monkeypatch, model_path):
    monkeypatch.setattr(main, "MODEL_PATH", model_path)
    with TestClient(main.app) as c:   # the `with` runs the startup (lifespan) hook
        yield c


@pytest.fixture
def client_without_model(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "MODEL_PATH", tmp_path / "does_not_exist.pkl")
    with TestClient(main.app) as c:
        yield c
