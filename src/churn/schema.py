# Autor: Alessandro Stefani
# Disciplina: Tech Challenge — Fase 01 (Produtização de Modelos)
# Descrição: Schema pandera do dataset cru — reutilizável (testes, validação, monitoramento).
"""Contrato de dados do dataset cru, em pandera.

Definido em `src/` (não nos testes) para ser **reutilizável**: o teste de schema importa
daqui, e a Etapa 4 usa o mesmo schema para validar entrada em batch e como base do
monitoramento de qualidade de dados.
"""
from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Check, Column


def schema_cru() -> pa.DataFrameSchema:
    """Tipos/faixas das colunas-chave do dataset cru.

    `ChurnLabel` é opcional (`required=False`) → o mesmo schema serve dados de treino
    (com alvo) e de inferência (sem alvo). `strict=False` não exige listar as 50 colunas.
    Nulos com significado (`Offer`, `InternetType`) são aceitos — imputados no pipeline.
    """
    return pa.DataFrameSchema(
        {
            "ChurnLabel": Column(str, Check.isin(["Yes", "No"]), required=False),
            "Age": Column(int, Check.in_range(18, 120)),
            "TenureinMonths": Column(int, Check.ge(0)),
            "MonthlyCharge": Column(float, Check.ge(0)),
            "TotalCharges": Column(float, Check.ge(0)),
            "Contract": Column(str, Check.isin(["Month-to-Month", "One Year", "Two Year"])),
            "Offer": Column(str, nullable=True),            # nulo = "No offer" (imputado)
            "InternetType": Column(str, nullable=True),     # nulo = "No internet" (imputado)
        },
        strict=False,
        coerce=True,
    )
