# Tech Challenge FIAP Machine Learning Engineering Fase 1
## Plano de Monitoramento (Churn Telco)


## 1. Por que monitorar (o risco real deste modelo)

A variável `tenure` domina o sinal e o dado é um **snapshot sem campanha de retenção**. Em produção, a base muda
(clientes novos, mudanças de preço/mix), as features derivam e o modelo degrada **em silêncio**, pois a métrica real aparece somente quando o churn acontece (semanas/meses depois). O monitoramento detecta a deriva **antes** do estrago acontecer.

## 2. O que já está pronto

| Peça | O que dá | Onde |
|---|---|---|
| `models/reference_profile.json` | linha de base (quantis/média/desvio das features + distribuições de proba) | gerado no `make train` |
| `src/churn/schema.py` | schema pandera (tipos/faixas/nulos com significado) para validação da entrada | reutilizável |
| Log de predição na API | `event=predict churn_proba=...` por request | **apenas stdout**|
| `threshold.json` | threshold operacional do artefato | usado pela decisão |

## 3. Fonte de dados para monitoramento: batch

O job periódico pontua a base inteira e constitui a **fonte oficial de monitoramento de drift**, por ser uma fotografia semelhante a usada nos treinamentos, sem duplicatas por cliente e com corte temporal bem definido.
Predições via **/predict (sob demanda) não são incluídas** no monitoramento de drift pelos seguintes motivos:
- O mesmo cliente pode ser consultado múltiplas vezes, **distorcendo as distribuições**
- Variáveis de valor (**TotalCharges, MonthlyCharge, TenureinMonths**) são sensíveis ao tempo
- O tráfego individual é **irregular e sem corte temporal controlado**
- Monitoramento via /predict seria considerado **somente na ausência de processo batch**, com **deduplicação obrigatória por CustomerID** (manter apenas o registro mais recente por cliente).

## 4. O que monitorar (3 níveis)

### 4.1 Data drift (entrada) — *leading indicator*
- **Por feature**, comparar janela atual × referência: **PSI** (Population Stability Index) e/ou **KS**.
- **Foco:** `tenure`, `MonthlyCharge`, `TotalCharges` (as que mais pesam).
- **Categóricas:** mudança de proporção + **taxa de categoria desconhecida** (reusar `schema.py`/`handle_unknown`).

### 4.2 Prediction drift (saída) — *barato e imediato*
- **Proba média** atual × referência (`reference_profile.json["proba_holdout"]`).
- **Taxa de churn previsto** (% acima do threshold) × esperado (~26–44% conforme o threshold).
- Detecta deslocamento mesmo **sem label**.

### 4.3 Performance drift (com label) — *ground truth, chega atrasado*
- Quando o churn real for conhecido, recalcular **PR-AUC / recall / custo** na janela.
- Comparar com o hold-out (PR-AUC ~0,78; custo ~$26,7k).

## 5. Playbook de resposta

1. **Alerta de data/prediction drift** - investigar a feature/categoria que disparou (mudança real de negócio? erro de coleta?).
2. **Reavaliar o threshold** com `evaluate.threshold_otimo` na janela recente.
3. **Retreino** (`make train`) - regenera `pipeline.joblib` + `threshold.json` + `reference_profile.json`
   (a referência passa a ser a nova) → reiniciar a API.
4. **Categoria nova frequente** -  avaliar inclusão no schema/encoder no próximo ciclo.

## 6. Cadência

- **Junto do batch:** para data e prediction drift.
- **Quando chegar label real:** para performance drift + decisão de retreino.
