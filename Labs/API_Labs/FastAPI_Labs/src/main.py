"""Iris species classifier served with FastAPI.

Run from this folder:   uvicorn main:app --reload
Docs:                   http://127.0.0.1:8000/docs
"""
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request, Response, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from predict import LoadedModel, ModelLoadError, load_model, predict
from schemas import (FEATURE_ORDER, BatchPredictionResponse, BatchRequest, ErrorResponse, IrisFeatures,
                     LegacyIrisResponse, ModelInfo, PredictionResponse)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("iris_api")

# Overridable so tests (or a deployment) can point at a different model file.
MODEL_PATH = Path(os.environ.get("MODEL_PATH", Path(__file__).resolve().parent.parent / "model" / "iris_model.pkl"))


# --------------------------------------------------------------------------------------------
# Startup: load the model ONCE (the original reloaded it from disk on every request)
# --------------------------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = load_model(MODEL_PATH, FEATURE_ORDER)
        app.state.model_error = None
        logger.info("Loaded model version %s from %s", app.state.model.version, MODEL_PATH)
    except ModelLoadError as exc:
        # Start anyway so /health and /ready can report the problem, instead of crash-looping.
        app.state.model, app.state.model_error = None, str(exc)
        logger.error("Model not loaded: %s", exc)
    yield


app = FastAPI(
    title="Iris Species Classifier API",
    version="1.0.0",
    description="Predicts the species of an iris flower from four measurements. "
                "Use the **v1** endpoints; `/predict` is kept only for old clients and is deprecated.",
    lifespan=lifespan,
    openapi_tags=[
        {"name": "v1", "description": "Current, versioned prediction API"},
        {"name": "operations", "description": "Liveness and readiness checks"},
        {"name": "legacy", "description": "The original lab's endpoint, kept for backward compatibility"},
    ],
)

ERROR_RESPONSES = {
    422: {"model": ErrorResponse, "description": "Invalid input"},
    503: {"model": ErrorResponse, "description": "Model not loaded"},
}


# --------------------------------------------------------------------------------------------
# Middleware: request id + timing on every response, one log line per request
# --------------------------------------------------------------------------------------------
@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
    request.state.request_id = request_id
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        # Unexpected errors: log the traceback server-side, never send it to the client.
        logger.exception("Unhandled error (request_id=%s)", request_id)
        response = JSONResponse(status_code=500, content=error_body(
            request, "internal_error", "Unexpected server error; see server logs with this request id."))
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time-ms"] = f"{elapsed_ms:.2f}"
    logger.info("%s %s -> %d in %.2f ms (request_id=%s)", request.method, request.url.path,
                response.status_code, elapsed_ms, request_id)
    return response


# --------------------------------------------------------------------------------------------
# One error format for everything
# --------------------------------------------------------------------------------------------
def error_body(request: Request, code: str, message: str, details=None):
    return {"error": {"code": code, "message": message, "details": details},
            "request_id": getattr(request.state, "request_id", None)}


@app.exception_handler(RequestValidationError)
async def validation_error_handler(request: Request, exc: RequestValidationError):
    details = [{"field": ".".join(str(part) for part in err["loc"]), "message": err["msg"], "type": err["type"]}
               for err in exc.errors()]
    return JSONResponse(status_code=422, content=error_body(request, "validation_error", "Request body is invalid",
                                                            details))


@app.exception_handler(StarletteHTTPException)
async def http_error_handler(request: Request, exc: StarletteHTTPException):
    codes = {404: "not_found", 405: "method_not_allowed", 503: "model_unavailable"}
    return JSONResponse(status_code=exc.status_code,
                        content=error_body(request, codes.get(exc.status_code, "http_error"), str(exc.detail)))


def get_model(request: Request) -> LoadedModel:
    """Dependency: the loaded model, or a 503 if it isn't available."""
    model = getattr(request.app.state, "model", None)
    if model is None:
        reason = getattr(request.app.state, "model_error", None) or "model not loaded"
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=f"Model unavailable: {reason}")
    return model


# --------------------------------------------------------------------------------------------
# Operations endpoints
# --------------------------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root():
    return {"service": app.title, "version": app.version, "docs": "/docs", "health": "/health", "ready": "/ready"}


@app.get("/health", tags=["operations"], summary="Liveness: the process is up")
def health():
    return {"status": "ok"}


@app.get("/ready", tags=["operations"], summary="Readiness: the model is loaded and can serve",
         responses={503: {"model": ErrorResponse, "description": "Model not loaded"}})
def ready(request: Request):
    model = getattr(request.app.state, "model", None)
    if model is None:
        reason = getattr(request.app.state, "model_error", None) or "model not loaded"
        return JSONResponse(status_code=503, content=error_body(request, "model_unavailable", reason))
    return {"status": "ready", "model_version": model.version}


# --------------------------------------------------------------------------------------------
# v1 API. Plain `def` (not `async def`): prediction is CPU-bound, so FastAPI runs these in its
# thread pool instead of blocking the event loop, as the original's `async def` did.
# --------------------------------------------------------------------------------------------
v1 = APIRouter(prefix="/v1", tags=["v1"], responses=ERROR_RESPONSES)


@v1.get("/model", response_model=ModelInfo, summary="Which model is being served")
def model_info(model: LoadedModel = Depends(get_model)):
    return ModelInfo(**{key: model.metadata[key] for key in ModelInfo.model_fields})


@v1.post("/predict", response_model=PredictionResponse, summary="Predict the species of one flower")
def predict_one(features: IrisFeatures, model: LoadedModel = Depends(get_model)):
    result = predict(model, [features.as_row()])[0]
    return PredictionResponse(**result, model_version=model.version)


@v1.post("/predict/batch", response_model=BatchPredictionResponse,
         summary="Predict the species of up to 100 flowers in one request")
def predict_batch(batch: BatchRequest, model: LoadedModel = Depends(get_model)):
    results = predict(model, [item.as_row() for item in batch.items])
    return BatchPredictionResponse(model_version=model.version, count=len(results), predictions=results)


app.include_router(v1)


# --------------------------------------------------------------------------------------------
# Legacy endpoint: same request and response shape as the original lab, marked deprecated
# --------------------------------------------------------------------------------------------
@app.post("/predict", response_model=LegacyIrisResponse, tags=["legacy"], deprecated=True,
          responses=ERROR_RESPONSES, summary="Deprecated: use POST /v1/predict")
def predict_legacy(features: IrisFeatures, response: Response, model: LoadedModel = Depends(get_model)):
    response.headers["Deprecation"] = "true"
    response.headers["Link"] = '</v1/predict>; rel="successor-version"'
    return LegacyIrisResponse(response=predict(model, [features.as_row()])[0]["class_id"])
