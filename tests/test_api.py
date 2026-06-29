# Autor: Alessandro Stefani
# Disciplina: Tech Challenge — Fase 01 (Produtização de Modelos)
# Descrição: Testes da API — /health 200, /predict proba∈[0,1], payload inválido → 422.
"""Testes da API FastAPI via TestClient.

O modelo é injetado via `dependency_overrides` (preditor tiny do conftest), então
os testes não dependem de um artefato treinado em disco.
"""
from __future__ import annotations

import json

from fastapi.testclient import TestClient

from churn import data
from churn.api.main import app, get_predictor


def _payload(amostra) -> dict:
    """Primeira linha crua como dict JSON-safe (NaN→null, numpy→nativo)."""
    X = data.selecionar_features(amostra)
    return json.loads(X.head(1).to_json(orient="records"))[0]


def test_health_ok():
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_proba_no_intervalo(predictor, amostra):
    app.dependency_overrides[get_predictor] = lambda: predictor
    try:
        client = TestClient(app)
        resp = client.post("/predict", json=_payload(amostra))
        assert resp.status_code == 200
        corpo = resp.json()
        assert 0.0 <= corpo["churn_probability"] <= 1.0
        assert isinstance(corpo["churn"], bool)
    finally:
        app.dependency_overrides.clear()


def test_predict_batch(predictor, amostra):
    app.dependency_overrides[get_predictor] = lambda: predictor
    try:
        client = TestClient(app)
        clientes = json.loads(data.selecionar_features(amostra).head(3).to_json(orient="records"))
        resp = client.post("/predict_batch", json={"clientes": clientes})
        assert resp.status_code == 200
        assert len(resp.json()["resultados"]) == 3
    finally:
        app.dependency_overrides.clear()


def test_payload_invalido_retorna_422(predictor):
    # modelo injetado p/ isolar a validação de entrada (sem isso o Depends falha antes, com 503)
    app.dependency_overrides[get_predictor] = lambda: predictor
    try:
        client = TestClient(app)
        resp = client.post("/predict", json={"Age": "não é número"})
        assert resp.status_code == 422
    finally:
        app.dependency_overrides.clear()
