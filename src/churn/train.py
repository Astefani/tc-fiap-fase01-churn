# Autor: Alessandro Stefani
# Disciplina: Tech Challenge — Fase 01 (Produtização de Modelos)
# Descrição: Treino do pipeline completo — CV + fit final + avaliação + MLflow + artefato.
"""Treina o Pipeline servível e registra no MLflow.

Fluxo: carrega → seleciona features → split dev/teste (80/20, estratificado) →
(opcional) CV estratificada no dev → fit final no dev → avalia no teste (1×) →
salva `models/pipeline.joblib` → loga params/métricas/modelo no MLflow.

Uso: `python -m churn.train`  (ou `make train`).
"""
from __future__ import annotations

import json
import logging

import joblib
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split

from . import config, data, evaluate
from .logging_config import configurar_logging
from .preprocess import construir_pipeline

logger = logging.getLogger(__name__)


def main(rodar_cv: bool = True, experimento: str = "telco_churn_mlp",
         registrar_mlflow: bool = True):
    """Treina, avalia, salva o artefato e registra no MLflow. Retorna (pipeline, métricas)."""
    configurar_logging()
    logger.info("Carregando dados: %s", config.DATA_RAW)
    df = data.carregar_dados()
    y = data.separar_alvo(df)
    X = data.selecionar_features(df, usar_geo=config.USAR_GEO)

    X_dev, X_te, y_dev, y_te = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.SEED)

    met = {}
    if rodar_cv:
        logger.info("Validação cruzada estratificada (%d folds)...", config.N_SPLITS)
        resumo = evaluate.validacao_cruzada(construir_pipeline(), X_dev, y_dev)
        met["cv_pr_auc_mean"], met["cv_pr_auc_std"] = resumo["test_pr_auc"]
        met["cv_dif_pr_auc"] = resumo["dif_pr_auc"][0]
        logger.info("CV PR-AUC (teste) = %.4f ± %.4f | dif_pr_auc = %.4f",
                    met["cv_pr_auc_mean"], met["cv_pr_auc_std"], met["cv_dif_pr_auc"])

    logger.info("Treino final no dev (%d linhas)...", len(X_dev))
    pipe = construir_pipeline()
    pipe.fit(X_dev, y_dev)

    proba_te = pipe.predict_proba(X_te)[:, 1]
    met.update(evaluate.calcular_metricas(y_te, proba_te))
    t_otimo, custo = evaluate.threshold_otimo(y_te, proba_te)
    met["threshold_otimo"], met["custo_otimo"] = t_otimo, custo
    logger.info("Hold-out: PR-AUC=%.4f | ROC-AUC=%.4f | recall=%.4f | t*=%.2f (custo $%d)",
                met["pr_auc"], met["roc_auc"], met["recall"], t_otimo, custo)

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, config.MODEL_PATH)
    logger.info("Pipeline salvo em %s", config.MODEL_PATH)

    # threshold operacional viaja com o artefato (lido pelo ChurnPredictor; config é fallback)
    config.THRESHOLD_PATH.write_text(json.dumps(
        {"threshold": t_otimo, "custo": custo,
         "custo_fn": config.CUSTO_FN, "custo_fp": config.CUSTO_FP}, indent=2))
    logger.info("Threshold ótimo salvo em %s (t*=%.3f)", config.THRESHOLD_PATH, t_otimo)

    # perfil de referência (linha de base p/ o monitoramento de drift da Etapa 4)
    perfil = evaluate.perfil_referencia(X_dev, proba_te)
    config.REFERENCE_PATH.write_text(json.dumps(perfil, indent=2, ensure_ascii=False))
    logger.info("Perfil de referência salvo em %s", config.REFERENCE_PATH)

    if registrar_mlflow:
        mlflow.set_experiment(experimento)
        with mlflow.start_run(run_name="pipeline_final"):
            mlflow.log_params({**config.MLP_CONFIG, "seed": config.SEED,
                               "usar_geo": config.USAR_GEO, "usar_contagens": config.USAR_CONTAGENS,
                               "n_splits": config.N_SPLITS,
                               "best_epoch": pipe.named_steps["model"].best_epoch_,
                               "dataset": config.DATASET_NOME})
            mlflow.log_metrics(met)
            ex = X_te.head(5)
            assinatura = infer_signature(ex, pipe.predict_proba(ex)[:, 1])
            mlflow.sklearn.log_model(pipe, "model", signature=assinatura, input_example=ex)
        logger.info("Run registrada no MLflow (experimento '%s')", experimento)

    return pipe, met


if __name__ == "__main__":
    main()
