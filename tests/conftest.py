# Autor: Alessandro Stefani
# Disciplina: Tech Challenge — Fase 01 (Produtização de Modelos)
# Descrição: Fixtures compartilhadas dos testes (amostra committed + pipeline tiny treinado).
"""Fixtures de teste.

Usa um fixture de dados **committed** (`tests/data/telco_sample.csv`, ~160 linhas)
— o dataset real é gitignored, então os testes não podem depender dele. O pipeline
é treinado pequeno e rápido (poucas épocas, rede mínima) só para validar o contrato.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from churn import data
from churn.predict import ChurnPredictor
from churn.preprocess import construir_pipeline

SAMPLE = Path(__file__).parent / "data" / "telco_sample.csv"
MLP_TINY = {"hidden": [16], "max_epochs": 5, "patience": 3}


@pytest.fixture(scope="session")
def amostra() -> pd.DataFrame:
    """Amostra crua committed (com `ChurnLabel`)."""
    return pd.read_csv(SAMPLE)


@pytest.fixture(scope="session")
def pipeline_treinado(amostra):
    """Pipeline completo treinado numa MLP mínima (rápido)."""
    y = data.separar_alvo(amostra)
    X = data.selecionar_features(amostra)
    pipe = construir_pipeline(mlp_config=MLP_TINY)
    pipe.fit(X, y)
    return pipe


@pytest.fixture(scope="session")
def predictor(pipeline_treinado) -> ChurnPredictor:
    return ChurnPredictor(pipeline_treinado, threshold=0.5)
