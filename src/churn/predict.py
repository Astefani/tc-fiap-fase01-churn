# Autor: Alessandro Stefani
# Disciplina: Tech Challenge — Fase 01 (Produtização de Modelos)
# Descrição: Serviço de inferência — serve batch (campanhas) e sob demanda do mesmo artefato.
"""`ChurnPredictor`: pontua clientes a partir do Pipeline treinado.

Dois modos de uso, **mesmo núcleo vetorizado** (`prever_proba`):
- **Batch** (seleção de campanhas massivas): `prever_lote` / `prever_lote_csv` /
  `python -m churn.predict --input ... --output ...`
- **Sob demanda** (na hora do contato com o cliente): `prever_um`, usado pela API.
"""
from __future__ import annotations

import argparse
import json
import logging

import joblib
import pandas as pd

from . import config
from .logging_config import configurar_logging

logger = logging.getLogger(__name__)


def _threshold_do_artefato() -> float:
    """Lê o threshold gravado no treino (`models/threshold.json`).

    Cai em `config.DECISION_THRESHOLD` se o arquivo não existir ou estiver inválido —
    assim o threshold servido acompanha o modelo servido, mas nunca falta um valor.
    """
    try:
        t = float(json.loads(config.THRESHOLD_PATH.read_text())["threshold"])
        logger.info("Threshold do artefato: %.4f (%s)", t, config.THRESHOLD_PATH.name)
        return t
    except (FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as e:
        logger.info("threshold.json indisponível (%s); fallback config=%.4f",
                    type(e).__name__, config.DECISION_THRESHOLD)
        return config.DECISION_THRESHOLD


class ChurnPredictor:
    """Carrega o Pipeline servível e pontua 1 ou N clientes crus."""

    def __init__(self, pipeline, threshold: float = config.DECISION_THRESHOLD):
        self.pipeline = pipeline
        self.threshold = threshold

    @classmethod
    def carregar(cls, caminho=None,
                 threshold: float | None = None) -> "ChurnPredictor":
        """Carrega o artefato (default: `config.MODEL_PATH`).

        `threshold=None` (padrão) → usa o salvo no treino (`models/threshold.json`),
        com fallback em `config.DECISION_THRESHOLD`. Passar um valor explícito sobrescreve.
        """
        caminho = caminho or config.MODEL_PATH
        logger.info("Carregando pipeline de %s", caminho)
        if threshold is None:
            threshold = _threshold_do_artefato()
        return cls(joblib.load(caminho), threshold=threshold)

    def prever_proba(self, X: pd.DataFrame):
        """Probabilidade de churn para N registros crus (núcleo dos dois modos)."""
        return self.pipeline.predict_proba(X)[:, 1]

    def prever_lote(self, df: pd.DataFrame) -> pd.DataFrame:
        """Modo BATCH: devolve o df de entrada + `churn_proba` + `churn_risco`."""
        out = df.copy()
        out["churn_proba"] = self.prever_proba(df)
        out["churn_risco"] = out["churn_proba"] >= self.threshold
        return out

    def prever_um(self, registro: dict) -> dict:
        """Modo SOB DEMANDA: 1 registro cru → probabilidade + decisão."""
        proba = float(self.prever_proba(pd.DataFrame([registro]))[0])
        return {"churn_probability": proba,
                "churn": bool(proba >= self.threshold),
                "threshold": self.threshold}


def prever_lote_csv(entrada: str, saida: str, caminho_modelo=None) -> pd.DataFrame:
    """Lê um CSV de clientes crus, pontua e grava o resultado."""
    preditor = ChurnPredictor.carregar(caminho_modelo)
    df = pd.read_csv(entrada)
    res = preditor.prever_lote(df)
    res.to_csv(saida, index=False)
    logger.info("Batch: %d registros pontuados → %s", len(res), saida)
    return res


def main(argv=None):
    """CLI de pontuação em lote."""
    configurar_logging()
    parser = argparse.ArgumentParser(description="Pontuação de churn em lote (batch).")
    parser.add_argument("--input", required=True, help="CSV de entrada (clientes crus)")
    parser.add_argument("--output", required=True, help="CSV de saída (com churn_proba/risco)")
    parser.add_argument("--model", default=None, help="caminho do pipeline.joblib")
    args = parser.parse_args(argv)
    prever_lote_csv(args.input, args.output, args.model)


if __name__ == "__main__":
    main()
