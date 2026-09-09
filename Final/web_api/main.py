"""FastAPI server connecting the Rocycle UI to PostgreSQL."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import FINAL_DIR, Settings
from .database import Database
from .statistics import CLASS_NAMES


LOGGER = logging.getLogger("rocycle.web_api")
settings = Settings.from_env()
database = Database(settings)


class ReviewCompleteRequest(BaseModel):
    final_class: Literal[
        "battery", "can", "paper", "pet_labeled", "plastic", "plastic_bag"
    ]
    reviewed_by: str | None = Field(default=None, max_length=100)


class ProcessingLogRequest(BaseModel):
    external_id: str | None = Field(default=None, max_length=120)
    predicted_class: Literal[
        "battery", "can", "paper", "pet_labeled", "plastic", "plastic_bag"
    ]
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    action: str | None = Field(default=None, max_length=50)
    destination: str | None = Field(default=None, max_length=50)
    result: str | None = Field(default=None, max_length=30)
    status: Literal["processing", "completed", "review_pending", "failed"] | None = None
    review_reason: str | None = Field(default=None, max_length=50)
    net_weight_kg: float | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.db_ready = False
    app.state.db_error = None
    try:
        database.initialize_schema()
        app.state.db_ready = True
    except Exception as exc:  # UI must still open and show a useful DB error.
        app.state.db_error = str(exc)
        LOGGER.exception("PostgreSQL schema initialization failed")
    yield


app = FastAPI(title="Rocycle UI API", version="1.0.0", lifespan=lifespan)


def db_call(operation):
    try:
        result = operation()
        app.state.db_ready = True
        app.state.db_error = None
        return result
    except ValueError:
        raise
    except Exception as exc:
        app.state.db_ready = False
        app.state.db_error = str(exc)
        raise HTTPException(status_code=503, detail="PostgreSQL 연결을 확인해주세요.") from exc


@app.get("/api/health")
def health():
    try:
        connected = database.ping()
        app.state.db_ready = connected
        app.state.db_error = None
    except Exception as exc:
        connected = False
        app.state.db_ready = False
        app.state.db_error = str(exc)
    return {
        "api": "ok",
        "database": "ok" if connected else "error",
        "checked_at": datetime.now().isoformat(),
        "error": app.state.db_error,
    }


@app.get("/api/dashboard")
def dashboard():
    return db_call(database.dashboard)


@app.get("/api/reviews")
def reviews():
    return {"items": db_call(database.list_reviews), "classes": list(CLASS_NAMES)}


@app.post("/api/reviews/{processing_log_id}/complete")
def complete_review(processing_log_id: int, request: ReviewCompleteRequest):
    try:
        result = db_call(
            lambda: database.complete_review(
                processing_log_id, request.final_class, request.reviewed_by
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=409, detail="이미 처리됐거나 검토 대기 항목이 아닙니다.")
    return result


@app.post("/api/processing", status_code=201)
def create_processing_log(request: ProcessingLogRequest):
    try:
        return db_call(lambda: database.create_processing_log(request.model_dump()))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/statistics")
def statistics(period: Literal["day", "week", "month"] = "day"):
    return db_call(lambda: database.statistics(period))


WEB_UI_DIR = FINAL_DIR / "web_ui"


@app.get("/", include_in_schema=False)
def ui_index():
    return FileResponse(WEB_UI_DIR / "index.html")


@app.get("/statistics.html", include_in_schema=False)
def ui_statistics():
    return FileResponse(WEB_UI_DIR / "statistics.html")


app.mount("/static", StaticFiles(directory=WEB_UI_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("web_api.main:app", host=settings.api_host, port=settings.api_port)
