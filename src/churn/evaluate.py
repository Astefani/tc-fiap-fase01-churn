# Autor: Alessandro Stefani
# Disciplina: Tech Challenge — Fase 01 (Produtização de Modelos)
# Descrição: Métricas, análise de custo (FP×FN) e validação cruzada estratificada.
"""Avaliação: métricas comparáveis, custo de negócio por threshold e CV estratificada.

A `validacao_cruzada` usa `cross_validate` sobre o Pipeline inteiro — como o
`TorchMLPClassifier` faz o sub-split de early stopping dentro do `fit`, a CV
"simplesmente funciona" também para a MLP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from . import config


def calcular_metricas(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Conjunto de métricas comparável ao dos notebooks (PR-AUC é a principal)."""
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_proba)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
    }


def custo_por_threshold(y_true, y_proba,
                        custo_fn: int = config.CUSTO_FN,
                        custo_fp: int = config.CUSTO_FP) -> pd.DataFrame:
    """Varre thresholds e calcula FP, FN e custo total = FP·custo_fp + FN·custo_fn."""
    y_true = np.asarray(y_true).ravel()
    y_proba = np.asarray(y_proba).ravel()
    linhas = []
    for t in np.linspace(0.05, 0.95, 181):
        pred = (y_proba >= t).astype(int)
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        linhas.append({"threshold": round(float(t), 4), "fp": fp, "fn": fn,
                       "custo_total": fp * custo_fp + fn * custo_fn})
    return pd.DataFrame(linhas)


def threshold_otimo(y_true, y_proba,
                    custo_fn: int = config.CUSTO_FN,
                    custo_fp: int = config.CUSTO_FP) -> tuple[float, int]:
    """Threshold que minimiza o custo total e o custo correspondente."""
    d = custo_por_threshold(y_true, y_proba, custo_fn, custo_fp)
    melhor = d.loc[d["custo_total"].idxmin()]
    return float(melhor["threshold"]), int(melhor["custo_total"])


def validacao_cruzada(pipeline, X, y,
                      n_splits: int = config.N_SPLITS, seed: int = config.SEED) -> dict:
    """StratifiedKFold CV do Pipeline. Retorna média ± desvio de PR-AUC/ROC-AUC.

    A MLP faz o split interno de early stopping em cada fold (via `cross_validate`,
    que clona e re-treina o pipeline por fold). `n_jobs=1` evita disputa pela GPU.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scoring = {"pr_auc": "average_precision", "roc_auc": "roc_auc"}
    res = cross_validate(pipeline, X, y, cv=skf, scoring=scoring,
                         return_train_score=True, n_jobs=1, error_score="raise")
    resumo = {chave: (float(np.mean(vals)), float(np.std(vals)))
              for chave, vals in res.items()
              if chave.startswith(("test_", "train_"))}
    resumo["dif_pr_auc"] = (resumo["train_pr_auc"][0] - resumo["test_pr_auc"][0], None)
    return resumo


def perfil_referencia(X, proba_holdout=None) -> dict:
    """Perfil estatístico das features de treino (+ distribuição de proba no hold-out).

    Salvo no treino (`train.py` → `models/reference_profile.json`) e usado na Etapa 4
    como linha de base para medir data/prediction drift (PSI/KS, taxa de categoria nova).
    """
    num = X.select_dtypes("number")
    cat = X.select_dtypes("object")
    perfil = {
        "n": int(len(X)),
        "numericas": {
            c: {"quantis": [float(v) for v in num[c].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1])],
                "media": float(num[c].mean()), "desvio": float(num[c].std())}
            for c in num.columns
        },
        "categoricas": {
            c: {str(k): round(float(v), 5)
                for k, v in cat[c].fillna("__NA__").value_counts(normalize=True).items()}
            for c in cat.columns
        },
    }
    if proba_holdout is not None:
        p = np.asarray(proba_holdout).ravel()
        perfil["proba_holdout"] = {
            "media": float(p.mean()),
            "quantis": [float(v) for v in np.quantile(p, [0, 0.25, 0.5, 0.75, 1])],
        }
    return perfil
