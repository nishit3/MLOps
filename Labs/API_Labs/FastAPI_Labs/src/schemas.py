"""Request and response schemas: the API's contract, enforced by Pydantic and shown in /docs."""
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field

# The order the model was trained on (checked against the model's metadata at startup).
FEATURE_ORDER = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
MAX_BATCH_SIZE = 100

SETOSA_EXAMPLE = {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2}


class IrisFeatures(BaseModel):
    """One flower's measurements in centimetres. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [SETOSA_EXAMPLE]})

    sepal_length: float = Field(..., gt=0, le=10, description="Sepal length in cm (0, 10]")
    sepal_width: float = Field(..., gt=0, le=10, description="Sepal width in cm (0, 10]")
    petal_length: float = Field(..., gt=0, le=10, description="Petal length in cm (0, 10]")
    petal_width: float = Field(..., gt=0, le=5, description="Petal width in cm (0, 5]")

    def as_row(self):
        return [getattr(self, name) for name in FEATURE_ORDER]


class BatchRequest(BaseModel):
    """Up to MAX_BATCH_SIZE flowers scored in one request."""

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [{"items": [
        SETOSA_EXAMPLE,
        {"sepal_length": 6.4, "sepal_width": 3.2, "petal_length": 4.5, "petal_width": 1.5},
        {"sepal_length": 6.3, "sepal_width": 3.3, "petal_length": 6.0, "petal_width": 2.5},
    ]}]})

    items: list[IrisFeatures] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class Prediction(BaseModel):
    class_id: int = Field(..., description="0 = setosa, 1 = versicolor, 2 = virginica")
    species: str
    probabilities: dict[str, float] = Field(..., description="Probability of each species")


class PredictionResponse(Prediction):
    model_config = ConfigDict(protected_namespaces=())
    model_version: str


class BatchPredictionResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_version: str
    count: int
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    model_version: str
    model_type: str
    trained_at: str
    test_accuracy: float
    features: list[str]
    classes: list[str]
    hyperparameters: dict


class LegacyIrisResponse(BaseModel):
    """The original lab's response shape, kept for the deprecated /predict route."""
    response: int


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[list[dict]] = None


class ErrorResponse(BaseModel):
    """Every error (422, 404, 503, 500) has this shape."""
    error: ErrorDetail
    request_id: Optional[str] = None
