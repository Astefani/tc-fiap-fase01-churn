# Autor: Alessandro Stefani
# Disciplina: Tech Challenge — Fase 01 (Produtização de Modelos)
# Descrição: Schemas Pydantic da API (entrada = features cruas; saída = probabilidade/decisão).
"""Contratos de entrada/saída da API.

`ClienteInput` reflete exatamente as features CRUS que o pipeline espera (as 31
colunas mantidas no NB01). `Offer` e `InternetType` são opcionais — o pipeline
imputa "No offer"/"No internet" quando ausentes.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ClienteInput(BaseModel):
    """Um cliente cru para inferência (mesmas colunas do dataset, sem o alvo)."""

    Gender: str
    Age: int = Field(ge=0)
    Married: str
    NumberofDependents: int = Field(ge=0)
    Latitude: float
    Longitude: float
    Population: int = Field(ge=0)
    Number_of_Referrals: int = Field(ge=0)
    TenureinMonths: int = Field(ge=1, description="meses como cliente (>=1)")
    Offer: str | None = None
    PhoneService: str
    AvgMonthlyLongDistanceCharges: float = Field(ge=0)
    MultipleLines: str
    InternetType: str | None = None
    AvgMonthlyGBDownload: int = Field(ge=0)
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtectionPlan: str
    PremiumTechSupport: str
    StreamingTV: str
    StreamingMovies: str
    StreamingMusic: str
    UnlimitedData: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharge: float = Field(ge=0)
    TotalCharges: float = Field(ge=0)
    TotalRefunds: float = Field(ge=0)
    TotalExtraDataCharges: int = Field(ge=0)
    TotalLongDistanceCharges: float = Field(ge=0)

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "Gender": "Male", "Age": 45, "Married": "Yes", "NumberofDependents": 0,
            "Latitude": 34.05, "Longitude": -118.24, "Population": 50000,
            "Number_of_Referrals": 2, "TenureinMonths": 24, "Offer": "Offer B",
            "PhoneService": "Yes", "AvgMonthlyLongDistanceCharges": 25.3,
            "MultipleLines": "No", "InternetType": "Fiber Optic", "AvgMonthlyGBDownload": 30,
            "OnlineSecurity": "No", "OnlineBackup": "Yes", "DeviceProtectionPlan": "No",
            "PremiumTechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes",
            "StreamingMusic": "No", "UnlimitedData": "Yes", "Contract": "Month-to-Month",
            "PaperlessBilling": "Yes", "PaymentMethod": "Credit Card", "MonthlyCharge": 89.5,
            "TotalCharges": 2100.5, "TotalRefunds": 0.0, "TotalExtraDataCharges": 0,
            "TotalLongDistanceCharges": 600.4,
        }
    })


class PredictResponse(BaseModel):
    """Resposta de inferência: probabilidade de churn + decisão no threshold operacional."""

    churn_probability: float = Field(ge=0, le=1)
    churn: bool
    threshold: float


class BatchRequest(BaseModel):
    """Lote de clientes para pontuação em massa (campanhas)."""

    clientes: list[ClienteInput]


class BatchResponse(BaseModel):
    resultados: list[PredictResponse]
