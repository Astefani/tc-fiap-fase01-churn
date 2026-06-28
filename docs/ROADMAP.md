# Roadmap de Desenvolvimento — Tech Challenge Fase 01

> Previsão de churn em telecom com rede neural (MLP/PyTorch), baselines (Scikit-Learn),
> tracking (MLflow) e API de inferência (FastAPI). Projeto end-to-end.

**Autor:** Alessandro Stefani
**Disciplina:** Tech Challenge — Fase 01 (Produtização de Modelos)
**Dataset:** Telco Customer Churn (IBM)

## Como este projeto evolui

O desenvolvimento segue a ordem em que as disciplinas da Fase 01 são cursadas.
Cada marco vira um ou poucos **commits significativos** — sem "tudo pronto" de uma vez.

- **Commits semânticos:** `feat:`, `fix:`, `chore:`, `docs:`, `test:`, `refactor:`
- **Fluxo Git:** trunk em `main` (projeto solo); branches por feature opcionais.
- **Reprodutibilidade desde o dia 1:** o projeto "instala do zero" via `poetry install`.

## Marcos

### Marco 0 — Bootstrap `chore`
Mínimo para começar a EDA e o baseline: `data/`, `notebooks/`, `pyproject.toml`
enxuto (pandas, scikit-learn, mlflow, jupyter), `.gitignore`, `README` esqueleto e
este roadmap. A estrutura de engenharia (`src/`, `tests/`, `Makefile`, ruff) entra
no Marco 3, junto da refatoração — como num fluxo real de quem começa pelo notebook.

### Marco 1 — Entendimento e Preparação · Etapa 1 (Disc. 01–02)
- [ ] ML Canvas (stakeholders, métricas de negócio, SLOs)
- [x] Download do dataset Telco para `data/raw/`
- [x] EDA completa (volume, qualidade, distribuição, data readiness)
- [x] Definir métrica técnica (AUC-ROC / PR-AUC / F1) e de negócio (custo de churn evitado)
- [x] Baselines: `DummyClassifier` + Regressão Logística
- [x] Primeiro tracking no MLflow (parâmetros, métricas, versão do dataset) — *refinar: faltam params + versão do dataset*

**Entregável:** notebook de EDA + baselines registrados no MLflow.

### Marco 2 — Rede Neural · Etapa 2 (Disc. 02) — *maior peso: 25%*
- [x] MLP em PyTorch (arquitetura, ativação, loss)
- [x] Loop de treino com early stopping e batching
- [x] Comparar MLP × baselines (≥ 4 métricas) — test set alinhado (64/16/20, mesmo 20% de teste que NB01)
- [x] Análise de trade-off de custo (falso positivo × falso negativo) — threshold ótimo ~0,27 (razão FN/FP 10:1)
- [x] Registrar todos os experimentos no MLflow

**Entregável:** tabela comparativa + MLP treinado + artefatos no MLflow.

> **Concluído (2026-06-29):** NB02 completo — MLP, OneHotEncoder, split alinhado com NB01,
> análise de custo FP×FN (threshold ótimo ~0,27), conclusões escritas. NB01 com RandomForest
> regularizado + LogReg balanceado + `dif_pr_auc`. Achado: modelos empatam ~0,76–0,78 PR-AUC;
> threshold importa mais que modelo (~35% economia vs ~5% entre modelos).
> Falta: atualizar markdowns do NB01 (números antigos) e ML Canvas (Marco 1).
> Plano das Etapas 3-4 em `docs/plano-etapas-3-4.md`.

### Marco 3 — Engenharia e API · Etapa 3 (Disc. 03–05)
- [ ] Refatorar do notebook para módulos em `src/` (estrutura limpa, SOLID)
- [ ] Pipeline reprodutível (sklearn + transformadores custom)
- [ ] Testes pytest: smoke, schema (pandera), API
- [ ] API FastAPI: `/predict`, `/health`, validação Pydantic
- [ ] Logging estruturado (sem `print()`) + middleware de latência
- [ ] `pyproject.toml`, ruff sem erros, Makefile (lint, test, run)

**Entregável:** repositório refatorado + API funcional + testes passando.

### Marco 4 — Documentação e Entrega · Etapa 4
- [ ] Model Card (performance, limitações, vieses, cenários de falha)
- [ ] Doc de arquitetura de deploy (batch × real-time + justificativa)
- [ ] Plano de monitoramento (métricas, alertas, playbook)
- [ ] README final (setup, execução, arquitetura)
- [ ] Vídeo de 5 min (método STAR)
- [ ] (Opcional/bônus) Deploy da API em nuvem

**Entregável:** repositório final + vídeo STAR + (opcional) URL do deploy.

## Pesos da avaliação (onde investir esforço)

| Peso | Critério |
|---|---|
| 25% | Rede neural (PyTorch) |
| 20% | Qualidade do código e estrutura |
| 15% | Pipeline e reprodutibilidade |
| 15% | API de inferência |
| 10% | Documentação e Model Card |
| 10% | Vídeo STAR |
| +5% | Bônus: deploy em nuvem |
