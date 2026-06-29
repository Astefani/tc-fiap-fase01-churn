# Autor: Alessandro Stefani
# Disciplina: Tech Challenge — Fase 01 (Produtização de Modelos)
# Descrição: Teste de schema (pandera) — o dataset cru satisfaz tipos/faixas/nulos com significado.
"""Valida o dataset cru contra um schema pandera.

Checa tipos e faixas das colunas-chave e reconhece os nulos COM significado
(`Offer`, `InternetType`) como válidos — eles são imputados no pipeline.
"""
from __future__ import annotations

from churn.schema import schema_cru


def test_dataset_cru_satisfaz_schema(amostra):
    schema_cru().validate(amostra)
