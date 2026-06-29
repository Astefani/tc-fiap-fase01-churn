# Autor: Alessandro Stefani
# Disciplina: Tech Challenge — Fase 01 (Produtização de Modelos)
# Descrição: Configuração central do pacote (single source of truth de constantes e caminhos).
"""Constantes, caminhos e hiperparâmetros do pacote `churn`.

As decisões de seleção/exclusão de features vêm do Notebook 01
(ver `docs/descricao-variaveis-dataset.md`). Centralizar aqui mantém os módulos
puros/injetáveis — sem globais escondidas.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Caminhos ──────────────────────────────────────────────────────────────
RAIZ = Path(__file__).resolve().parents[2]              # .../fase-01-tc
DATA_RAW = RAIZ / "data" / "raw" / "TelcoCustomerChurn.csv"
MODELS_DIR = RAIZ / "models"
# artefato servível único (batch + API); sobrescrevível por env (deploy)
MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH") or (MODELS_DIR / "pipeline.joblib"))
REFERENCE_PATH = MODELS_DIR / "reference_profile.json"   # perfil de referência p/ drift (Etapa 4)
THRESHOLD_PATH = MODELS_DIR / "threshold.json"           # threshold ótimo do artefato (do treino)

# ── Reprodutibilidade ───────────────────────────────────────────────────────
SEED = 42

# ── Alvo / dataset ──────────────────────────────────────────────────────────
TARGET = "ChurnLabel"
TARGET_MAP = {"No": 0, "Yes": 1}
DATASET_NOME = "TelcoCustomerChurn.csv"

# ── Seleção de features (decisões do NB01) ──────────────────────────────────
COLS_DROP = [
    TARGET, "CustomerID",
    "CustomerStatus", "ChurnScore", "ChurnCategory", "ChurnReason",   # leakage
    "CLTV",                                                            # precaução
    "ZipCode", "SeniorCitizen", "Under30", "TotalRevenue",            # redundância
    "InternetService", "City",                                        # redundância / cardinalidade
    "Country", "State", "Quarter",                                    # constantes
    "SatisfactionScore",                                              # leakage confirmado
    "Dependents", "ReferredaFriend",                                 # flags colineares → contínuas
]
COLS_GEO = ["Latitude", "Longitude"]   # removidas quando USAR_GEO=False

# ── Imputação de negócio (missing com significado, leakage-safe) ────────────
IMPUTE_VALUES = {"Offer": "No offer", "InternetType": "No internet"}

# ── Engenharia de features ──────────────────────────────────────────────────
USAR_GEO = True
USAR_CONTAGENS = False
SKEWED = ["TotalLongDistanceCharges", "AvgMonthlyGBDownload"]   # cauda direita → log1p

# ── MLP (config vencedora por CV, corrigida no Mac 2026-06-29 — ver docs/ROADMAP.md) ──
# A CV sobre o treino de 64% preferiu a rede menor (mais parcimoniosa; dif_pr_auc ~0,037,
# metade do overfit da rede maior). Vencedor: mlp_ba64 → [64, 32], batch 64.
MLP_CONFIG = {
    "hidden": [64, 32],
    "dropout": 0.3,
    "lr": 1e-3,
    "batch": 64,
    "max_epochs": 200,
    "patience": 20,
    "val_size": 0.2,        # sub-split interno p/ early stopping
}

# ── Validação cruzada estratificada ─────────────────────────────────────────
N_SPLITS = 5
TEST_SIZE = 0.2            # hold-out final (mesmo 20% dos notebooks)

# ── Custo de negócio (FP × FN) e threshold operacional ──────────────────────
CUSTO_FN = 500            # perder um cliente (receita futura)
CUSTO_FP = 50             # campanha de retenção desnecessária  → razão FN/FP 10:1
# threshold de decisão p/ campanhas (abaixo de 0,5 por causa do custo assimétrico).
# FALLBACK: o threshold operacional vem de `models/threshold.json` (gravado pelo `train.py`
# com o `threshold_otimo` do hold-out, que viaja com o artefato). Este valor só é usado
# quando o arquivo não existe — o ótimo varia a cada retreino (modelo diferente, calibração
# diferente), por isso o número certo é o salvo no treino, não esta constante.
DECISION_THRESHOLD = 0.135

# ── API / logging ───────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("CHURN_LOG_LEVEL", "INFO")
API_TITULO = "Churn Telecom — API de Inferência"
API_VERSAO = "0.1.0"
