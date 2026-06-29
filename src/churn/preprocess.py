# Autor: Alessandro Stefani
# Disciplina: Tech Challenge — Fase 01 (Produtização de Modelos)
# Descrição: Pipeline reprodutível (sklearn + transformador custom) — o coração da Etapa 3.
"""Monta o Pipeline serializável que recebe dados CRUS e prediz churn.

FeatureEngineer → ColumnTransformer (log1p+scale nas skewed, scale nas demais
numéricas, one-hot nas categóricas) → TorchMLPClassifier.

Salvo com `joblib`, este único artefato serve **batch** (campanhas) e **sob demanda**
(API) — o `OneHotEncoder(handle_unknown="ignore")` ajustado é o que faz um registro
único funcionar igual ao lote.
"""
from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from . import config
from .features import FeatureEngineer
from .model import TorchMLPClassifier


def _num_nao_skewed(X):
    """Numéricas exceto as skewed (estas levam log1p antes do scale)."""
    return [c for c in X.select_dtypes("number").columns if c not in config.SKEWED]


def construir_preprocessador() -> ColumnTransformer:
    """ColumnTransformer: log1p+scale (skewed) | scale (num) | one-hot (cat)."""
    skew_pipe = Pipeline([
        ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("skew", skew_pipe, list(config.SKEWED)),
            ("num", StandardScaler(), _num_nao_skewed),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False),
             make_column_selector(dtype_include=object)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def construir_pipeline(mlp_config=None, usar_contagens: bool = config.USAR_CONTAGENS) -> Pipeline:
    """Pipeline completo (cru → predição). `mlp_config` sobrescreve `config.MLP_CONFIG`."""
    cfg = {**config.MLP_CONFIG, **(mlp_config or {})}
    return Pipeline([
        ("features", FeatureEngineer(usar_contagens=usar_contagens)),
        ("preprocess", construir_preprocessador()),
        ("model", TorchMLPClassifier(
            hidden=cfg["hidden"], dropout=cfg["dropout"], lr=cfg["lr"], batch=cfg["batch"],
            max_epochs=cfg["max_epochs"], patience=cfg["patience"],
            val_size=cfg["val_size"], seed=config.SEED)),
    ])
