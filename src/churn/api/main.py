# Autor: Alessandro Stefani
# Disciplina: Tech Challenge — Fase 01 (Produtização de Modelos)
# Descrição: API FastAPI de inferência — /health, /predict (sob demanda), /predict_batch (lote).
"""API de inferência de churn.

Endpoints:
- `GET /health` — liveness + se o modelo está carregado.
- `POST /predict` — 1 cliente (sob demanda, contato com o cliente).
- `POST /predict_batch` — lote (seleção de campanhas massivas).

Os dois endpoints de inferência chamam o mesmo núcleo (`ChurnPredictor`). Logging
estruturado + middleware de latência (header `X-Process-Time-ms`).

Subir: `uvicorn churn.api.main:app --reload` (ou `make api`).
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse

from .. import config
from ..logging_config import configurar_logging
from ..predict import ChurnPredictor
from . import schemas

configurar_logging()
logger = logging.getLogger("churn.api")

_predictor: ChurnPredictor | None = None


def _carregar_predictor() -> ChurnPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ChurnPredictor.carregar()
    return _predictor


def get_predictor() -> ChurnPredictor:
    """Dependência FastAPI (injetável/sobrescrevível em testes)."""
    try:
        return _carregar_predictor()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503,
                            detail="Modelo não disponível — treine com `make train`.") from e


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _carregar_predictor()
        logger.info("Modelo carregado no startup (%s)", config.MODEL_PATH)
    except Exception as e:  # noqa: BLE001 — startup não deve derrubar a app
        logger.warning("Modelo não carregado no startup: %s", e)
    yield


app = FastAPI(title=config.API_TITULO, version=config.API_VERSAO, lifespan=lifespan)


@app.middleware("http")
async def medir_latencia(request: Request, call_next):
    inicio = time.perf_counter()
    resposta = await call_next(request)
    dur_ms = (time.perf_counter() - inicio) * 1000
    resposta.headers["X-Process-Time-ms"] = f"{dur_ms:.1f}"
    logger.info("method=%s path=%s status=%s latency_ms=%.1f",
                request.method, request.url.path, resposta.status_code, dur_ms)
    return resposta


@app.get("/", include_in_schema=False)
def root():
    """Raiz: redireciona para a documentação interativa (Swagger UI)."""
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok", "modelo_carregado": _predictor is not None}


@app.post("/predict", response_model=schemas.PredictResponse)
def predict(cliente: schemas.ClienteInput, preditor: ChurnPredictor = Depends(get_predictor)):
    """Sob demanda: pontua 1 cliente na hora do contato."""
    resultado = preditor.prever_um(cliente.model_dump())
    logger.info("event=predict churn_proba=%.4f churn=%s threshold=%.2f",
                resultado["churn_probability"], resultado["churn"], resultado["threshold"])
    return resultado


@app.post("/predict_batch", response_model=schemas.BatchResponse)
def predict_batch(req: schemas.BatchRequest, preditor: ChurnPredictor = Depends(get_predictor)):
    """Batch: pontua um lote de clientes (campanhas massivas)."""
    df = pd.DataFrame([c.model_dump() for c in req.clientes])
    probas = preditor.prever_proba(df)
    logger.info("event=predict_batch n=%d churn_proba_media=%.4f",
                len(probas), float(probas.mean()))
    resultados = [
        schemas.PredictResponse(churn_probability=float(p),
                                churn=bool(p >= preditor.threshold),
                                threshold=preditor.threshold)
        for p in probas
    ]
    return schemas.BatchResponse(resultados=resultados)
