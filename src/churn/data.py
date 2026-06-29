# Autor: Alessandro Stefani
# Disciplina: Tech Challenge — Fase 01 (Produtização de Modelos)
# Descrição: Carregamento e seleção de dados crus (transformações ficam no pipeline).
"""Acesso a dados: carregar o CSV, separar o alvo e selecionar as features cruas.

Estas funções NÃO transformam os dados (sem fillna/encode/scale) — isso é
responsabilidade do pipeline (`preprocess.construir_pipeline`), para que o mesmo
artefato sirva treino, batch e API a partir de registros crus.
"""
from __future__ import annotations

import pandas as pd

from . import config


def carregar_dados(caminho=None) -> pd.DataFrame:
    """Lê o dataset bruto (default: `config.DATA_RAW`)."""
    return pd.read_csv(caminho or config.DATA_RAW)


def separar_alvo(df: pd.DataFrame) -> pd.Series:
    """Mapeia o alvo `ChurnLabel` (No/Yes) → 0/1."""
    return df[config.TARGET].map(config.TARGET_MAP)


def selecionar_features(df: pd.DataFrame, usar_geo: bool = config.USAR_GEO) -> pd.DataFrame:
    """Remove colunas de leakage/redundância/constantes (decisões do NB01).

    `usar_geo=False` também descarta Latitude/Longitude. Robusto a colunas ausentes
    (ex.: registro de inferência sem `ChurnLabel`).
    """
    cols_drop = list(config.COLS_DROP)
    if not usar_geo:
        cols_drop += config.COLS_GEO
    return df.drop(columns=[c for c in cols_drop if c in df.columns])
