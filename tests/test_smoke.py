# Autor: Alessandro Stefani
# Disciplina: Tech Challenge — Fase 01 (Produtização de Modelos)
# Descrição: Smoke test — o pipeline treina e prediz sem erro, com saída no shape/range esperado.
"""Smoke test do pipeline e do preditor (batch + sob demanda)."""
from __future__ import annotations

import numpy as np

from churn import data
from churn.preprocess import construir_pipeline


def test_pipeline_treina_e_preve(amostra):
    """Pipeline cru→predição: shape (n,) e probabilidades em [0, 1]."""
    y = data.separar_alvo(amostra)
    X = data.selecionar_features(amostra)
    pipe = construir_pipeline(mlp_config={"hidden": [16], "max_epochs": 3, "patience": 2})
    pipe.fit(X, y)

    proba = pipe.predict_proba(X.head(5))[:, 1]
    assert proba.shape == (5,)
    assert np.all((proba >= 0) & (proba <= 1))


def test_registro_unico_funciona(predictor, amostra):
    """Sob demanda: um único registro cru produz probabilidade válida."""
    X = data.selecionar_features(amostra)
    resp = predictor.prever_um(X.iloc[0].to_dict())
    assert set(resp) == {"churn_probability", "churn", "threshold"}
    assert 0.0 <= resp["churn_probability"] <= 1.0
    assert isinstance(resp["churn"], bool)


def test_lote_funciona(predictor, amostra):
    """Batch: pontua N registros e devolve as colunas de saída."""
    X = data.selecionar_features(amostra)
    out = predictor.prever_lote(X.head(10))
    assert {"churn_proba", "churn_risco"} <= set(out.columns)
    assert len(out) == 10
    assert out["churn_proba"].between(0, 1).all()
