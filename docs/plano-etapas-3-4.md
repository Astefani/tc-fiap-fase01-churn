# Plano — Etapas 3 e 4 (Tech Challenge Fase 01)

> Planejamento da refatoração (Etapa 3) e da documentação/entrega (Etapa 4).
> Complementa o `ROADMAP.md` (Marcos 3 e 4). Baseado no enunciado em
> `docs/Tech Challenge Fase 01.pdf`.

## Pré-requisitos (fechar a Etapa 2 antes)
1. ~~**Alinhar o split** — mesmo test set nos NB01 e NB02.~~ ✅
2. ~~**Trade-off de custo FP×FN** (threshold pelo custo, igual p/ todos).~~ ✅
3. ~~**Validação cruzada estratificada dirigindo a seleção**~~ ✅ — `notebooks/02-modelo-mlp-cv.ipynb`
   (`StratifiedKFold`, 5 folds, **sobre o treino de 64%**; melhor candidato pela média de PR-AUC;
   hold-out reportado 1×). Vencedor: **`mlp_ba64`** (`[64,32]`, batch 64 — CV 0,7676 ± 0,0122;
   hold-out PR-AUC 0,7805; `dif_pr_auc` 0,037). A CV com dados limpos preferiu a rede menor, o que
   **resolve a regra do 1-desvio**. **Pendente:** espelhar a mesma CV no NB01.
4. Atualizar markdowns do NB01 (números balanceados). *(pendente)*
5. ML Canvas (item pendente do Marco 1). *(pendente)*

---

## Etapa 3 — Engenharia e API (Disciplinas 03–05)
**Entregável:** repositório refatorado + API funcional + testes passando.
**Critérios que pesam:** qualidade/estrutura do código (20%), pipeline/reprodutibilidade (15%), API (15%).

> **STATUS (2026-06-29): Etapa 3 CONCLUÍDA (Fases A–E).**
> Detalhes completos em **`diversos/etapa3-implementacao.md`**. Decisões tomadas:
> (a) **MLP como estimador sklearn** (`TorchMLPClassifier`) num **Pipeline único** — serve batch +
> sob demanda do mesmo `models/pipeline.joblib`; (b) **CV no `evaluate.py`** via `cross_validate`
> (funciona com o wrapper); (c) imputação de negócio movida para dentro do pipeline (`FeatureEngineer`).
> **Fase E concluída (Ubuntu, 2026-06-29):** `pyproject` ajustado (packages + deps +
> `uvicorn[standard]` + ruff E/F/I + pytest); **ruff limpo** e **pytest 8/8**; pacote `churn`
> instalado. **`make train` executado** → `pipeline.joblib` + `reference_profile.json` +
> `threshold.json` + run MLflow. Extras: rota `/` → `/docs` na API; **threshold operacional viaja
> com o artefato** (`threshold.json`, fallback em `config`). **Falta só o commit no Mac.**

### Fase A — Refatorar notebooks → `src/churn/` (módulos) ✅
**Implementado** — ver `diversos/etapa3-implementacao.md`. (A tabela abaixo é o *plano*; a
implementação real difere em pontos menores: a imputação de negócio foi para o `FeatureEngineer`;
`evaluate.py` tem `validacao_cruzada` em vez de `comparar_modelos`; adicionados `logging_config.py`
e `model.TorchMLPClassifier`.)

| Módulo | O que implementar | Vem de |
|---|---|---|
| `config.py` | SEED, paths, COLS_DROP, MLP_CONFIG, grupos de colunas, IMPUTE_VALUES | NB01/NB02 |
| `data.py` | `carregar_dados`, `imputar_negocio`, `separar_alvo`, `selecionar_features(usar_geo)` | Seção 2 dos NBs |
| `features.py` | `FeatureEngineer` (transformador custom): `valor_medio_fatura` + contagens | Seção 3 NB02 |
| `preprocess.py` | `construir_pipeline`: FeatureEngineer → ColumnTransformer (log1p+scale skewed, scale num, OneHot cat) | Seção 4 NB02 |
| `model.py` | `MLP` | Seção 5 NB02 |
| `train.py` | `treinar_mlp` (early stopping) + `main()` (treina + loga MLflow) | Seção 6 NB02 |
| `evaluate.py` | `calcular_metricas`, `comparar_modelos`, `custo_negocio` (FP×FN) | Seção 7 NB02 |
| `predict.py` | `ChurnPredictor` (serve 1 ou N) + `prever_lote` (batch CLI) | novo (já desenhado) |

**Princípio:** funções **puras/injetáveis** (sem globais escondidas — lição da Seção 7);
constantes vêm do `config.py`.

### Fase B — Pipeline reprodutível (sklearn + transformadores custom) ✅
- O **Pipeline serializável** (`FeatureEngineer` → `ColumnTransformer` → modelo) é o coração:
  recebe dados **crus** e faz fillna→features→log1p→one-hot→scale→predict.
- Salvar o Pipeline ajustado junto do modelo no MLflow → serve **batch e caso-a-caso** do mesmo
  artefato (o `OneHotEncoder` ajustado é o que faz registro único funcionar).
- Requisito do PDF: "pipeline reprodutível (sklearn + transformadores custom)".

### Fase C — Testes (pytest, ≥ 3) ✅ *(escritos; rodam após a Fase E — pandera/pytest faltam instalar)*
- `test_smoke.py` — pipeline treina e prediz sem erro; saída no shape/range esperado.
- `test_schema.py` — **pandera**: o dataset cru satisfaz o schema (tipos, faixas, nulos com significado).
- `test_api.py` — `/health` 200; `/predict` proba∈[0,1]; payload inválido → 422.

### Fase D — API FastAPI ✅
- `GET /health`, `POST /predict` (1 cliente, Pydantic `ClienteInput`), `POST /predict_batch` (lote).
- **Logging estruturado** (sem `print()`) + **middleware de latência**.
- Os dois endpoints chamam o mesmo núcleo (`ChurnPredictor.prever_proba`, vetorizado).

### Fase E — Tooling / pyproject / ruff / Makefile ✅
- **Dependências novas** (sugestão: grupos): `fastapi`, `uvicorn`, `pydantic`, `pandera`,
  `pytest`, `ruff`, `httpx` (TestClient).
- **pyproject:** trocar `package-mode = false` por `packages = [{include = "churn", from = "src"}]`
  (senão `python -m churn.train` / `uvicorn churn.api.main` não acham o pacote) + config do ruff.
- **Makefile** (já criado): `install`, `lint`, `test`, `train`, `api`, `clean`.
- **ruff sem erros** (critério).

**Ordem sugerida da Etapa 3:** A (config→data→features→preprocess→model) → B (pipeline) →
train/evaluate/predict → C (testes) → D (API) → E (tooling). Commitar por fase (histórico limpo).
**Progresso: Etapa 3 concluída — A–E ✅ + `make train` executado (2026-06-29). Falta só o commit no Mac.**

---

## Etapa 4 — Documentação e Entrega Final
**Entregável:** repositório final + vídeo STAR + (opcional) deploy.
**Critérios:** documentação/model card (10%), vídeo STAR (10%), bônus deploy (5%).

### Model Card (`docs/notas-para-model-card.md` → versão final)
- **Performance:** tabela comparativa (Dummy × LogReg × RF × MLP) + curvas ROC/PR.
- **Limitações:** modelos empatam ~0,76 (modelo importa pouco); **tenure domina** o dataset;
  **sem campanha de retenção** nos dados (modelo cego a essa alavanca); snapshot transversal.
- **Vieses / data leakage:** `SatisfactionScore` (excluída); `Offer` ≈ cohort de tenure.
- **Overfit honesto:** usar o `dif_pr_auc` (RF decora mais que LogReg/MLP).
- **Cenários de falha:** drift de tenure/preço; categoria nova (handle_unknown cobre); cliente novo.

### Arquitetura de deploy (batch vs real-time + justificativa)
- O `ChurnPredictor` serve os **dois**. Recomendação a documentar: **batch noturno** como deploy
  principal (churn não exige latência de ms; pontuar a base periodicamente) + **API real-time**
  (exigida) para consulta caso-a-caso. Justificar o trade-off.

### Plano de monitoramento
- Métricas: **drift de PR-AUC** (quando houver label), **data drift** (distribuição das features,
  esp. tenure/charges), taxa de churn prevista vs real.
- Alertas + playbook de resposta (retreino, investigação).

### README final
- Setup (`poetry install` — torch cross-machine cu128/MPS), execução (`make ...`), arquitetura.

### Vídeo 5 min (método STAR)
- **S**ituation: problema de churn + contexto do dataset.
- **T**ask: tarefa do grupo + objetivos técnicos.
- **A**ction: decisões (features, leakage, métrica PR-AUC, MLP × baselines, regularização, deploy).
- **R**esult: resultados (empate ~0,76, dif_pr_auc, leakage) + lições aprendidas.

### (Opcional, bônus) Deploy em nuvem
- API FastAPI com endpoint público (AWS/Azure/GCP).

---

## Sequência macro recomendada
Fechar Etapa 2 (split + custo + revisão) → Etapa 3 (refatorar em fases, commitando) →
Etapa 4 (docs + model card + vídeo). ML Canvas pode entrar em paralelo.
