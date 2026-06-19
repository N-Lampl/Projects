"""OPTIONAL enhanced transport: a FastAPI app over the same PredictionService.

FastAPI is imported lazily inside `create_app()` so this module imports fine even
when FastAPI/uvicorn are not installed (the default path uses stdlib_server instead).
Run with:  make serve-fastapi   (needs `pip install fastapi uvicorn`).

NOTE: no `from __future__ import annotations` here — FastAPI needs the route's type
hints to be real objects (e.g. `Request`), not stringified, to build its signature.
"""

from .service import PredictionService

API_KEY_HEADER = "x-api-key"


def create_app(service=None, **build_kwargs):
    """Build a FastAPI app. Raises ImportError with guidance if FastAPI is missing."""
    try:
        from fastapi import FastAPI, Header, Request
        from fastapi.responses import JSONResponse
    except ImportError as e:  # pragma: no cover - exercised only when fastapi absent
        raise ImportError(
            "FastAPI not installed. Use the default stdlib path (`make serve`/`make demo`) "
            "or `pip install fastapi uvicorn` for this optional transport."
        ) from e

    svc = service or PredictionService.build(**build_kwargs)
    app = FastAPI(title="api-threat-model", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/predict")
    async def predict(request: Request, x_api_key: str = Header(default=None)):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(status_code=400, content={"error": "invalid JSON"})
        resp = svc.handle(x_api_key, "/predict", body)
        return JSONResponse(status_code=resp.status, content=resp.body)

    # expose for tests / introspection
    app.state.service = svc
    return app
