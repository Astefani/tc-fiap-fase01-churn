# Autor: Alessandro Stefani
# Disciplina: Tech Challenge — Fase 01 (Produtização de Modelos)
# Descrição: Transformador custom de engenharia de features (1º passo do pipeline).
"""`FeatureEngineer`: imputação de negócio + features derivadas.

Recebe os dados CRUS (colunas selecionadas) e devolve um DataFrame com:
- imputação de negócio (`Offer`, `InternetType`) — missing com significado, leakage-safe;
- `valor_medio_fatura` = TotalCharges / TenureinMonths (razão, não redundante);
- (opcional) contagens de serviços adicionais.

É *stateless* (não aprende nada relevante no `fit`), então funciona igual para 1 ou
N registros — é o que faz o serving caso-a-caso casar com o treino.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from . import config


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Imputa `Offer`/`InternetType` e cria `valor_medio_fatura` (+ contagens opcionais)."""

    def __init__(self, usar_contagens: bool = config.USAR_CONTAGENS, impute_values=None):
        self.usar_contagens = usar_contagens
        self.impute_values = impute_values        # None → usa config.IMPUTE_VALUES

    def fit(self, X, y=None):
        self.impute_values_ = dict(self.impute_values or config.IMPUTE_VALUES)
        self.feature_names_in_ = np.asarray(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # imputação de negócio (missing = ausência do serviço/oferta)
        for col, valor in self.impute_values_.items():
            if col in X.columns:
                X[col] = X[col].fillna(valor)

        # razão: valor médio histórico da fatura (tenure mínimo = 1 → sem div/0)
        X["valor_medio_fatura"] = X["TotalCharges"] / X["TenureinMonths"]

        if self.usar_contagens:
            def sim(coluna):
                return X[coluna].eq("Yes").astype(int)

            X["qtd_adicionais"] = (sim("OnlineSecurity") + sim("OnlineBackup")
                                   + sim("DeviceProtectionPlan") + sim("PremiumTechSupport"))
            X["qtd_streaming"] = sim("StreamingTV") + sim("StreamingMovies") + sim("StreamingMusic")
            X["qtd_principais"] = (sim("PhoneService")
                                   + X["InternetType"].ne("No internet").astype(int))
        return X
