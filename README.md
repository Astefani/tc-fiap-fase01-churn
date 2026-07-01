# Churn Telecom — Tech Challenge Fase 01

Previsão de churn de clientes em uma operadora de telecomunicações, do dado cru ao modelo servido via API.
Projeto desenvolvido para o Tech Challenge da Fase 01 da Pós Tech ML Engineering (FIAP + Alura).

**Autor:** Alessandro Stefani  
**Modelo central:** rede neural MLP (PyTorch), comparada a baselines Scikit-Learn, rastreada com MLflow e servida via FastAPI.

---

## Resultados

| Modelo | PR-AUC | ROC-AUC | dif_pr_auc | recall @ t\* | custo @ t\* |
|---|---|---|---|---|---|
| Dummy (prevalência) | 0,272 | 0,516 | — | — | — |
| Regressão Logística | 0,764 | 0,904 | 0,010 | 0,965 | $26.800 |
| Random Forest | 0,747 | 0,898 | 0,093 | 0,963 | $27.500 |
| **MLP [64,32] (artefato servido)** | **0,783** | **0,906** | — | **0,957** | **$26.700** |

Avaliação no hold-out de 20% (1.409 clientes), com threshold operacional otimizado por custo (FN×$500 / FP×$50).

**Achado central:** os três modelos não-triviais empatam em ~0,75–0,78 PR-AUC. Ajustar o threshold de 0,5 para o ótimo reduz o custo em ~30% — mais impactante que a escolha do modelo. A MLP foi escolhida por produtização e parcimônia (menor `dif_pr_auc` que o RF, mesmo desempenho da LogReg com mais flexibilidade).

---

## Stack

Python 3.12 · PyTorch · Scikit-Learn · MLflow · FastAPI · Pydantic · Pandera · Pytest · Ruff · Poetry

---

## Estrutura

```
.
├── data/
│   └── raw/                        # dataset bruto (não versionado — ver §Setup)
├── notebooks/
│   ├── 01-eda-baseline.ipynb       # EDA + baselines + MLflow
│   └── 02-modelo-mlp-cv.ipynb      # MLP com seleção por validação cruzada
├── src/churn/                      # pacote Python
│   ├── config.py                   # constantes e caminhos (single source of truth)
│   ├── data.py                     # carregamento e seleção de features
│   ├── features.py                 # FeatureEngineer (transformador custom)
│   ├── preprocess.py               # Pipeline sklearn reprodutível
│   ├── model.py                    # MLP (nn.Module) + TorchMLPClassifier (wrapper sklearn)
│   ├── train.py                    # treino: CV + fit + avaliação + MLflow + artefato
│   ├── evaluate.py                 # métricas, custo FP×FN, validação cruzada, perfil de referência
│   ├── predict.py                  # ChurnPredictor (batch + sob demanda) + CLI
│   ├── schema.py                   # schema Pandera (validação de dados crus)
│   ├── logging_config.py           # logging estruturado (sem print())
│   └── api/
│       ├── main.py                 # FastAPI: /health, /predict, /predict_batch
│       └── schemas.py              # Pydantic: ClienteInput, PredictResponse, Batch*
├── tests/
│   ├── conftest.py                 # fixtures (amostra, pipeline tiny, predictor)
│   ├── data/telco_sample.csv       # amostra comprometida (testes independentes do dataset bruto)
│   ├── test_smoke.py               # pipeline treina + prediz; predictor (dois modos)
│   ├── test_schema.py              # Pandera valida o dataset cru
│   └── test_api.py                 # /health 200; /predict ∈[0,1]; payload inválido → 422
├── models/                         # artefatos gerados por `make train` (não versionados)
│   ├── pipeline.joblib             # Pipeline sklearn completo (FeatureEngineer → ColumnTransformer → MLP)
│   ├── threshold.json              # threshold operacional ótimo (minimiza custo FP×FN)
│   └── reference_profile.json     # distribuições de referência para monitoramento de drift
├── docs/                           # documentação do projeto
├── Makefile                        # targets: install, lint, test, train, api, clean
└── pyproject.toml                  # dependências, ruff, pytest (único source of truth)
```

---

## Setup

### Pré-requisitos

- Python 3.12
- [Poetry](https://python-poetry.org/docs/#installation)
- GPU com CUDA 12.8 (opcional — o treinamento também roda na CPU)

### 1. Instalar dependências

```bash
make install
# equivale a: poetry install
```

### 2. Baixar o dataset

O dataset **Telco Customer Churn (IBM)** está disponível publicamente no Kaggle:

```
https://www.kaggle.com/datasets/yeanzc/telco-customer-churn-ibm-dataset
```

Baixe o arquivo `Telco_customer_churn.csv` e coloque em `data/raw/TelcoCustomerChurn.csv`.

O arquivo não é versionado (`.gitignore`) por convenção de projetos de ML.

---

## Como rodar

### Lint

```bash
make lint
# ruff check src tests — deve retornar "All checks passed!"
```

### Testes

```bash
make test
# pytest -q — 8 testes (smoke, schema Pandera, API)
```

Os testes usam uma amostra comprometida (`tests/data/telco_sample.csv`) e não dependem do dataset bruto.

### Treino

```bash
make train
# python -m churn.train
```

Gera três artefatos em `models/`:
- `pipeline.joblib` — Pipeline completo (pré-processamento + MLP)
- `threshold.json` — threshold operacional ótimo
- `reference_profile.json` — perfil de referência para monitoramento de drift

Registra o experimento `telco_churn_mlp` no MLflow (pasta `mlruns/`).

```bash
# Para visualizar os experimentos no MLflow UI:
poetry run mlflow ui
# Acesse http://localhost:5000
```

### API

```bash
make api
# uvicorn churn.api.main:app --reload
# Acesse http://localhost:8000/docs
```

#### Endpoints

| Método | Endpoint | Descrição |
|---|---|---|
| GET | `/health` | Liveness + status do modelo |
| POST | `/predict` | Previsão para 1 cliente |
| POST | `/predict_batch` | Previsão para N clientes |

#### Exemplo — previsão de 1 cliente

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Gender": "Male",
    "Age": 45,
    "Married": "Yes",
    "NumberofDependents": 0,
    "Latitude": 34.05,
    "Longitude": -118.24,
    "Population": 50000,
    "Number_of_Referrals": 2,
    "TenureinMonths": 24,
    "Offer": "Offer B",
    "PhoneService": "Yes",
    "AvgMonthlyLongDistanceCharges": 25.3,
    "MultipleLines": "No",
    "InternetType": "Fiber Optic",
    "AvgMonthlyGBDownload": 30,
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtectionPlan": "No",
    "PremiumTechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "StreamingMusic": "No",
    "UnlimitedData": "Yes",
    "Contract": "Month-to-Month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Credit Card",
    "MonthlyCharge": 89.5,
    "TotalCharges": 2100.5,
    "TotalRefunds": 0.0,
    "TotalExtraDataCharges": 0,
    "TotalLongDistanceCharges": 600.4
  }'
```

Resposta:

```json
{
  "churn_probability": 0.72,
  "churn": true,
  "threshold": 0.135
}
```

### Batch (CSV)

```bash
python -m churn.predict \
  --input data/raw/clientes.csv \
  --output pontuacao.csv
```

O CSV de saída é o de entrada acrescido das colunas `churn_probability` e `churn`.

---

## Arquitetura

### Um artefato, dois modos

O coração do sistema é um **Pipeline sklearn único** que recebe dados crus (31 features) e devolve a probabilidade de churn. Ele é salvo em `models/pipeline.joblib` e serve tanto o modo batch quanto a API real-time.

```
dados crus (31 features)
        │
        ▼
┌───────────────────────────────────────────────────────────┐
│                   models/pipeline.joblib                  │
│                                                           │
│  1. FeatureEngineer (custom)                              │
│     • imputa Offer/InternetType (ausência tem significado)│
│     • cria valor_medio_fatura = TotalCharges / Tenure     │
│                                                           │
│  2. ColumnTransformer                                     │
│     • skewed: log1p → StandardScaler                      │
│     • numéricas: StandardScaler                           │
│     • categóricas: OneHotEncoder(handle_unknown="ignore") │
│                                                           │
│  3. TorchMLPClassifier                                    │
│     • MLP [64, 32], ReLU, Dropout 0,3                     │
│     • sigmoid(logit) = P(churn)                           │
└───────────────────────────────────────────────────────────┘
        │
        ▼
  P(churn) ∈ [0,1]
  churn = P > threshold (lido de models/threshold.json)
```

### Dois modos de entrega

| | Batch (principal) | Real-time (API) |
|---|---|---|
| Caso de uso | campanhas de retenção | atendimento caso-a-caso |
| Latência | irrelevante (job periódico) | baixa (ms) |
| Volume | base inteira | 1 ou N por request |
| Custo de infra | mínimo (roda e termina) | servidor sempre no ar |
| Interface | CLI (`python -m churn.predict`) | FastAPI (`POST /predict`) |

Ambos os modos usam o mesmo `pipeline.joblib`, garantindo consistência entre canais.

### Fluxo de treino

```
CSV → carregamento → seleção de features → split 80/20 (dev/holdout, estratificado, SEED=42)
    → CV estratificada 5-folds no dev (seleção de hiperparâmetros por PR-AUC)
    → fit do pipeline no dev (MLP com early stopping em val PR-AUC interno, patience=20)
    → avaliação no holdout (1×, nunca visto)
    → salva pipeline.joblib + threshold.json + reference_profile.json
    → loga no MLflow (params + métricas + artefato com signature)
```

---

## Documentação

| Arquivo | Conteúdo |
|---|---|
| [docs/model-card.md](docs/model-card.md) | Model Card: performance, limitações, vieses, cenários de falha |
| [docs/model-card-aux.md](docs/model-card-aux.md) | Seleção e exclusão de features: justificativas técnicas por variável |
| [docs/deploy.md](docs/deploy.md) | Arquitetura de deploy: batch vs. real-time, operação, rollback |
| [docs/monitoramento.md](docs/monitoramento.md) | Plano de monitoramento: métricas, alertas, playbook |
| [docs/dicionario-de-dados.md](docs/dicionario-de-dados.md) | Dicionário de dados e justificativas de exclusão de features |
| [docs/Tech Challenge Fase 01.pdf](docs/Tech%20Challenge%20Fase%2001.pdf) | Enunciado oficial do Tech Challenge |


