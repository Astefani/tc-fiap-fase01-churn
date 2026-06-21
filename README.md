# Churn Telecom — Tech Challenge Fase 01

Previsão de churn em telecom com rede neural (MLP/PyTorch), comparada a baselines
(Scikit-Learn), rastreada com MLflow e servida via API (FastAPI).

> **Status:** 🚧 em desenvolvimento — ver [docs/ROADMAP.md](docs/ROADMAP.md).

**Autor:** Alessandro Stefani
**Curso:** Pós Tech ML Engineering (FIAP) — Fase 01 (Produtização de Modelos)
**Dataset:** Telco Customer Churn (IBM)

## Estrutura

O projeto começa enxuto (foco na EDA e no baseline) e ganha estrutura de
engenharia (`src/`, testes, API) na etapa de refatoração — ver [docs/ROADMAP.md](docs/ROADMAP.md).

```
.
├── data/raw/    # dataset bruto (não versionado)
├── notebooks/   # EDA e experimentação
└── docs/        # ROADMAP e documentação
```

## Setup

```bash
poetry install      # instala dependências (Python 3.12)
poetry run jupyter lab
```

<!-- TODO: completar conforme o projeto evolui -->
- [ ] Como baixar o dataset
- [ ] Como rodar a EDA / treinar os modelos
- [ ] Como subir a API (`/predict`, `/health`)
- [ ] Como rodar os testes e o lint
- [ ] Arquitetura de deploy e plano de monitoramento
