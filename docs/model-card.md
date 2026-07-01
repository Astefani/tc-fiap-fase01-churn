# Tech Challenge FIAP Machine Learning Engineering Fase 1
## Model Card (Churn Telco)
  
## Modelo

**Nome:** Churn Telco — MLP (PyTorch) em Pipeline sklearn servível  
**Versão:** 0.1.0  
**Data:** 2026-06-29  
**Autor:** Alessandro Stefani  
**Repositório:** https://github.com/Astefani/tc-fiap-fase01-churn  
**Tipo:** classificação binária (churn: sim/não)
**Saída:** probabilidade ∈ [0,1] + decisão no threshold operacional
**Artefato:** `models/pipeline.joblib` (`FeatureEngineer → ColumnTransformer → TorchMLPClassifier`)

## Uso pretendido

- **Batch** (deploy principal): pontuar a base periodicamente para **seleção de campanhas de retenção**.
- **API real-time** (`POST /predict`): consulta caso-a-caso **na hora do contato** com o cliente.

## Dados

- **Telco Customer Churn (IBM)** — 7.043 clientes.
- **Prevalência de churn:** 26,5% (classe desbalanceada).
- **Limites do dado:** não há registro de campanha de retenção, então o modelo é **cego a essa alavanca**;
  é uma foto (sem série temporal).
- **Split:** 80/20 estratificado (hold-out de 20% = 1.409 clientes); seleção de hiperparâmetros por
  **CV estratificada (5 folds) sobre o treino**; hold-out reportado 1×.

## Features

- **31 mantidas** (de 50): descartadas por leakage, redundância, constância ou cardinalidade (decisões do NB01).
- **`tenure` (meses de casa) domina** o sinal.
- **Imputação de negócio (leakage-safe):** `Offer`→"No offer", `InternetType`→"No internet" (ausência tem significado).
- **Transformações:** `log1p` nas caudas (`TotalLongDistanceCharges`, `AvgMonthlyGBDownload`); `StandardScaler`;
  `OneHotEncoder(handle_unknown="ignore")` (cobre categoria nova em produção).
- **Feature derivada:** `valor_medio_fatura = TotalCharges / TenureinMonths`.

## Métrica

- **Principal: PR-AUC** (average precision) — adequada a classe rara.
- **Decisão por custo FP×FN:** `CUSTO_FN=500` (perder cliente) × `CUSTO_FP=50` (campanha desnecessária) → **razão 10:1**.
  O threshold operacional minimiza o custo total (≠ 0,5) e **é servido junto com o artefato** (`models/threshold.json`).

## Treinamento

- **Arquitetura:** rede com duas camadas `[64, 32]`, ativação ReLU e Dropout 0,3 entre camadas, com saída em 1 logit.
- **Loss:** `BCEWithLogitsLoss` com `pos_weight` calculado dinamicamente (~2,77×, razão negativo/positivo no sub-treino) para tratar o desbalanceamento.
- **Otimizador:** Adam, lr 1e-3.
- **Early stopping:** monitora PR-AUC no conjunto de validação, com sub-split interno de 20%), patience de 20 épocas e máximo 200 épocas.
- **Artefato servido:** treinado no dev (80% do dataset), com máximo de 200 épocas e parada antecipada se PR-AUC no val não melhorar por 20 épocas consecutivas.

## Performance

### Seleção por CV (treino de 64%, 5 folds) — candidatos MLP
Melhor média: `mlp_ba64` (0,7658 ± 0,0128); banda de 1σ ≥ 0,7531 → **todos os 8 candidatos empatam**.
Pela **regra do 1-desvio** (mais simples dentro de 1σ): arquitetura **`[64,32]`, lr 1e-3** (CV PR-AUC 0,7634 ± 0,0127).

> **Produção usa batch 64** (`config.MLP_CONFIG`). Dentro do conjunto `[64,32]` empatado em 1σ, o batch
> não muda a arquitetura nem a performance → mantido 64 (decisão registrada). O candidato base do notebook
> (`mlp_padrao`) usava 256; é o mesmo modelo para efeitos práticos.

### Hold-out (teste 20%, 1.409 clientes) — comparação entre modelos

Métricas independentes de threshold (PR-AUC, ROC-AUC) + custo no threshold ótimo de cada modelo.
Todos **sem** a feature de leakage; baselines (NB01) e MLP (NB02/servido) compartilham o **mesmo 20% de
teste** (splits alinhados).

| Modelo | PR-AUC | ROC-AUC | dif_pr_auc | recall @ t* | custo @ t* |
|---|---|---|---|---|---|
| Dummy (prevalência) | 0,272 | 0,516 | ~0 | — | — |
| LogReg | 0,764 | 0,904 | 0,010 | 0,965 (@0,25) | $26.800 |
| RandomForest | 0,747 | 0,898 | **0,093** | 0,963 (@0,305) | $27.500 |
| MLP — notebook (seleção) | 0,776 | 0,910 | 0,041 | 0,976 (@0,195) | $26.150 |
| **MLP — artefato servido** | **0,783** | **0,906** | †~0,08 | 0,957 (@0,135) | $26.700 |

† O gap do servido é medido contra o set de treino de 80% (definição diferente do notebook, treino de
64%) → não comparável 1:1; o indicador limpo é o do notebook (0,041). Ver §Overfit.

> **Achado-chave:** fora o Dummy, os modelos **empatam ~0,75–0,78 PR-AUC** (custos a ~$1–2k de distância)
> → **o modelo importa pouco; o threshold importa muito** (custo cai ~30% de 0,5 para o ótimo). Pontuar a
> base e mirar a lista sinalizada dá **lift ~2,16×** sobre a taxa-base de churn (26,5%).

### Modelo servido vs. notebook — por que os números diferem

O **modelo oficial é o artefato servido** (`models/pipeline.joblib`) (que roda em batch e na API).
O **notebook `02-modelo-mlp-cv.ipynb` é a origem da seleção** (CV + regra do 1-desvio elegeram a
arquitetura `[64,32]`). Os números **não são idênticos** porque o `make train` não reproduz bit a bit o refit do notebook:
- **Early stopping:** `make train` recorta um val interno *aleatório* do treino; o notebook usa o `X_val` fixo.
- **Refit:** notebook treina no `X_tr` (64%); o serving treina no dev (80%) com val interno.
- **Batch:** produção usa **64** (o candidato base do notebook usava 256) — irrelevante (empate em 1σ).
- **Não-determinância de GPU.**

Mesma arquitetura, mesmo 20% de hold-out → PR-AUC na mesma faixa (0,776 vs 0,783), não idêntica.
**Reportamos o servido**; o notebook é a base metodológica da escolha.

## Limitações

- Empate entre modelos (a escolha da MLP é por produtização/parcimônia, não por ganho de métrica).
- `tenure` domina → sensível a mudanças de mix de base (ex.: onda de clientes novos).
- O dataset é uma fotografia dos dados em um momento específico, sem dinâmica temporal e sem efeito de campanha de retenção nos dados.

## Vieses e data leakage

- **`SatisfactionScore` excluída** — leakage confirmado: **com** ela LogReg/RF chegam a PR-AUC
  **0,981 / 0,972** (ROC ~0,99) e **sem** ela caem para **0,764 / 0,747**. A feature praticamente
  "entrega" o desfecho, por isso foi excluída do pipeline.
- Colunas `Churn*` (Score/Category/Reason) e `CustomerStatus` = o próprio desfecho → descartadas.
- `Offer` se mostrou depender do momento em que o cliente entrou na base (correlacionada com tenure), mas foi mantida (conscientemente).
- `CLTV` é uma estimativa da receita relacionada ao tempo de permanência do cliente. Por não sabermos a lógica de criação dela, foi descartada por precaução (possível vazamento de valor futuro).

## Overfit

- Indicador `dif_pr_auc` (train − test):  
**LogReg 0,010**  
**MLP `[64,32]` 0,041**  
**RandomForest 0,093**  
  
  O RF **decora ~2× mais** que a MLP (e ~9× mais que a LogReg) para ganho de PR-AUC **nulo**.
- O gap do *artefato servido* medido contra os 80% de treino é ~0,08 (maior **por definição** — train set
  maior, não pior generalização); o indicador limpo e comparável é o do notebook/CV acima.

## Cenários de falha e mitigação

- **Drift de tenure/preço** - monitorar distribuição.
- **Categoria nova** - coberta por `handle_unknown="ignore"`. Monitorar taxa de "desconhecido".
- **Cliente novíssimo** (tenure baixo) - predição menos confiável.
- **Mudança de mix de produtos** - reavaliar threshold (`threshold_otimo`) e considerar retreino.

## Reprodutibilidade

- `poetry install` (torch cross-machine cu128/MPS)  
- `make train` regenera `pipeline.joblib` +
  `reference_profile.json` + `threshold.json` + run MLflow. SEED=42.

---
