# Tech Challenge FIAP Machine Learning Engineering Fase 1
## Arquitetura de Deploy (Churn Telco)

  
  
## 1. Princípio: um artefato, dois modos

Toda a inferência sai do **mesmo** `models/pipeline.joblib` — um `Pipeline` sklearn serializável que
recebe dados **crus** e devolve probabilidade de churn:

```
                            models/pipeline.joblib
        ┌───────────────────────────────────────────────────────────┐
        │  FeatureEngineer → ColumnTransformer → TorchMLPClassifier │
        │  (imputação +      (log1p+scale,        (MLP PyTorch,     │
        │   valor_medio)      OneHot, scale)        CPU no serving) │
        └───────────────────────────────────────────────────────────┘
                    ▲                                 ▲
         dados crus (CSV / JSON)              dados crus (JSON)
                    │                                 │
       ┌────────────┴────────────┐     ┌──────────────┴──────────────┐
       │ MODO BATCH (principal)  │     │   MODO REAL-TIME (exigido)  │
       │ python -m churn.predict │     │   FastAPI POST /predict     │
       │ --input ... --output... │     │   (1 cliente, no contato)   │
       └─────────────────────────┘     └─────────────────────────────┘
```

- O `OneHotEncoder` **ajustado** é o que faz "1 registro" funcionar igual ao lote (colunas consistentes).
- **Threshold operacional** lido de `models/threshold.json` (viaja com o artefato; `config` é fallback).
- **Treina na GPU, serve na CPU** → artefato portável (Ubuntu cu128 / Mac MPS / qualquer CPU).

## 2. Os dois modos

### 2.1 Batch (deploy principal)
- **Para:** pontuar a base inteira periodicamente (ex.: job noturno).
- **Como:** `python -m churn.predict --input data/raw/clientes.csv --output pontuacao.csv`.
- **Por que é o principal:** Campanhas de marketing são planejadas e não demandam baixa latência, então um job noturno (barato) atende a necessidade.
- **Contrato:** CSV com as 31 features cruas (mesmas colunas do ClienteInput);
- **Saída:** CSV de entrada, com o acréscimo da probabilidade e da flag de churn.

### 2.2 Real-time / sob demanda
- **Para:** pontuar **1 cliente na hora** do atendimento ou um lote pequeno.
- **Como:** `make api`.
- **Por que existe:** Caso o atendimento/retenção precise do score **no momento do contato** com o cliente.
- **Contrato:** Pydantic `ClienteInput`.
- **Saída:** JSON { "churn_probability": float, "churn": bool, "threshold": float }.

## 3. Trade-off (a justificativa pedida)

| Critério | Batch | Real-time (API) |
|---|---|---|
| Latência | irrelevante (job) | baixa (ms) |
| Volume | base inteira de uma vez | 1 ou poucos por request |
| Custo de infra | mínimo (roda e morre) | servidor sempre no ar |
| Caso de uso | campanhas massivas | contato individual |
| Frescor do score | do último job | sob demanda |

**Decisão:** **batch como deploy principal** (maior valor, menor custo) e **API real-time** para o
caso-a-caso (exigência + valor no atendimento).  
Ambos consomem o mesmo artefato, garantindo consistência entre canais.

## 4. Operação

- **Logging:** estruturado em **stdout**.
- **Latência:** middleware adiciona header `X-Process-Time-ms` e loga `latency_ms` por request.
- **Health:** `GET /health` (liveness + se o modelo carregou).
- **Sem modelo em disco:** API retorna **503** ao receber a primeira requisição (o predictor não carrega no startup se o artefato não existir).

## 5. Troca de modelo / rollback

- O caminho do artefato é definido em variável de ambiente: `CHURN_MODEL_PATH=/path/novo.joblib`, sem alterar o código.
- Fluxo de retreino: `make train` sobrescreve `pipeline.joblib`, `threshold.json` e `reference_profile.json`. Depois é necessário reiniciar a API manualmente (ela carregará o artefato no startup).
- O modelo antigo **não é preservado automaticamente**. Para ter rollback, é necessário versionar o artefato antes de retreinar (ex.: `cp models/pipeline.joblib models/pipeline_v1.joblib`) ou resgatar via MLflow, juntamente com o `threshold_otimo` (registrado em Model metrics).
- Para apontar para um artefato específico (versão anterior ou resgatado do MLflow): criar um `threshold.json` com o valor resgatado do MLflow + `CHURN_MODEL_PATH=/path/para/pipeline_v1.joblib` + restart da API.

## 6. (Bônus) Deploy em nuvem

Endpoint público da API (FastAPI) — opções rápidas sem container (containers são Fase 02):
Render / Railway / Fly.io / Cloud Run. Opcional (+5%).

---
