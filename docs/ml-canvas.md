# ML Canvas — Churn Telecom (TC Fase 01)

**Autor:** Alessandro Stefani  
**Disciplina:** Tech Challenge — Fase 01 (Produtização de Modelos)

---

## Problema de negócio

Uma operadora de telecomunicações enfrenta perda acelerada de clientes. Sem visibilidade
antecipada sobre quem está em risco de cancelamento, as ações de retenção chegam tarde ou
são disparadas de forma indiscriminada — gerando custo desnecessário ou perdendo clientes que
poderiam ter sido retidos.

**Pergunta central:** Dado o perfil atual de um cliente, qual a probabilidade de ele cancelar
o serviço até o fim do trimestre?

---

## Stakeholders

| Stakeholder | Papel | Como usa o modelo |
|---|---|---|
| Time de Marketing | Planeja campanhas de retenção | Consome o score em **batch** para selecionar a lista de clientes a abordar |
| Time de Atendimento | Contato direto com o cliente | Consulta o score em **tempo real** (API) no momento do atendimento |
| Gestão / Diretoria | Define estratégia e aprova budget | Acompanha KPIs de churn evitado e custo das campanhas |
| Time de Dados / ML | Mantém e evolui o modelo | Opera retreino, monitora drift, avalia degradação |

---

## Dados

- **Dataset:** Telco Customer Churn (IBM) — disponível publicamente no Kaggle.
- **Volume:** 7.043 clientes × 50 colunas originais.
- **Alvo:** `ChurnLabel` (Yes/No → 1/0). Prevalência de churn: **~26,5%** (desbalanceado).
- **Natureza:** snapshot transversal (fotografia de um trimestre). Sem série temporal,
  sem registro de campanhas de retenção anteriores.
- **Features usadas:** 31 de 50 — 14 numéricas, 17 categóricas. Remoções por leakage,
  redundância, constância e alta cardinalidade (detalhes em `docs/model-card-aux.md`).

---

## Predições

| Item | Detalhe |
|---|---|
| **O que o modelo prediz** | Probabilidade de churn do cliente ∈ [0, 1] |
| **Granularidade** | 1 cliente por predição (pode processar N em batch) |
| **Frequência** | Batch periódico (ex.: noturno) + sob demanda via API |
| **Latência esperada** | Batch: tolerante (minutos) · API: < 200 ms |

---

## Features

**Preditores mais relevantes (EDA + coeficientes LogReg):**

- `TenureinMonths` — dominante (correlação -0,35 com churn; clientes novos churnam mais)
- `Contract` — Month-to-Month com 45,8% de churn vs 2,6% em Two Year
- `MonthlyCharge` — quanto maior a mensalidade, maior o risco
- `Offer` — Offer E tem 52,9% de churn; Offer A tem 6,7%
- `InternetType` — Fiber Optic com 40,7% de churn vs DSL 18,6%
- `OnlineSecurity` / `PremiumTechSupport` — clientes sem serviços adicionais churnam mais
- `PaperlessBilling` — associada a maior churn (33,6% vs 16,3%)

**Feature derivada:** `valor_medio_fatura = TotalCharges / TenureinMonths`

**Tratamentos:** imputação de negócio (`Offer` → "No offer", `InternetType` → "No internet");
`log1p` em features de cauda direita; `StandardScaler`; `OneHotEncoder(handle_unknown="ignore")`.

---

## Modelo

| Item | Detalhe |
|---|---|
| **Tipo** | Classificação binária |
| **Arquitetura central** | MLP PyTorch — camadas `[64, 32]`, ReLU, Dropout 0,3 |
| **Loss** | `BCEWithLogitsLoss` com `pos_weight` (~2,77×) para tratar desbalanceamento |
| **Otimizador** | Adam, lr 1e-3 |
| **Early stopping** | Monitora PR-AUC no val interno (patience 20, máx 200 épocas) |
| **Seleção** | CV estratificada 5-folds no dev (64% do dataset); vencedor por PR-AUC médio |
| **Baselines** | DummyClassifier, Regressão Logística, Random Forest (todos via sklearn) |
| **Artefato** | Pipeline sklearn serializável: `FeatureEngineer → ColumnTransformer → MLP` |

---

## Métricas técnicas

| Métrica | Valor (hold-out 20%) | Por quê |
|---|---|---|
| **PR-AUC** (principal) | **0,783** | Robusta a classe positiva rara (~26,5%) |
| ROC-AUC | 0,906 | Complementar |
| Recall @ t* | 0,957 | Captura a maioria dos churners |
| dif_pr_auc | ~0,04 | Indicador de overfit (baixo) |

**Comparação com baselines (hold-out):**

| Modelo | PR-AUC | dif_pr_auc |
|---|---|---|
| Dummy | 0,272 | — |
| Regressão Logística | 0,764 | 0,010 |
| Random Forest | 0,747 | 0,093 |
| **MLP [64,32]** | **0,783** | 0,041 |

---

## Métricas de negócio

**Custo por erro:**
- **Falso Negativo** (churner não identificado): cliente cancela sem ação → **$500**
- **Falso Positivo** (alarme falso): campanha de retenção desnecessária → **$50**
- Razão FN/FP: **10:1**

**Função de custo:** `custo = FP × $50 + FN × $500`

| Modelo | Threshold ótimo | Custo (hold-out) |
|---|---|---|
| Regressão Logística | 0,25 | $26.800 |
| Random Forest | 0,305 | $27.500 |
| **MLP (artefato servido)** | **0,135** | **$26.700** |

**Achado central:** ajustar o threshold de 0,5 para o ótimo (~0,135) reduz o custo em ~30%.
A escolha do modelo impacta apenas 3–5% — o threshold importa mais.

---

## Decisões

- O score de probabilidade é transformado em decisão binária (churn / não churn) pelo
  **threshold operacional** (0,135), escolhido para minimizar o custo FP×FN.
- **Batch:** clientes com `churn = True` entram na lista da campanha de retenção.
- **API:** score retornado em tempo real para o atendente decidir o nível de intervenção.
- O threshold viaja com o artefato (`models/threshold.json`) e pode ser ajustado sem retreino
  se o custo relativo FP/FN mudar.

---

## SLOs (Service Level Objectives)

| Indicador | Alvo mínimo | Ação se violar |
|---|---|---|
| PR-AUC no hold-out | ≥ 0,75 | Retreino obrigatório |
| Recall @ threshold ótimo | ≥ 0,85 | Revisar threshold |
| Custo total (hold-out) | ≤ $30.000 | Alerta; revisar threshold ou retreinar |
| Latência da API (`/predict`) | < 200 ms (p99) | Investigar gargalo |
| Taxa de categoria desconhecida | < 5% | Investigar drift de features categóricas |

---

## Deploy e operação

- **Modo principal:** batch periódico — pontua a base inteira, alimenta campanhas de retenção.
- **Modo complementar:** API real-time (`POST /predict`) — consulta no momento do atendimento.
- **Artefato único:** `models/pipeline.joblib` serve ambos os modos (consistência garantida).
- **Retreino:** `make train` regenera pipeline + threshold + perfil de referência.
- **Rollback:** versionar o `.joblib` antes de retreinar ou resgatar via MLflow.

---

## Monitoramento

- **Data drift:** PSI/KS em `tenure`, `MonthlyCharge`, `TotalCharges` contra `reference_profile.json`.
- **Prediction drift:** Δ na taxa de churn previsto e na probabilidade média.
- **Performance drift:** recalcular PR-AUC/recall quando os labels reais ficarem disponíveis.
- **Fonte:** output do job batch (snapshot homogêneo, sem duplicatas por cliente).
- **Cadência:** junto do batch para data/prediction drift; mensal para performance drift.

---

## Limitações e riscos

- `tenure` domina o sinal → sensível a ondas de clientes novos (mudança de mix de base).
- Dataset sem registro de campanhas de retenção → modelo cego a esse efeito.
- Foto de um único trimestre → sem captura de sazonalidade ou tendências de longo prazo.
- `SatisfactionScore` excluída por leakage confirmado (PR-AUC 0,79 → 0,98 com ela) —
  se disponível em produção antes do evento, reavaliar inclusão.
