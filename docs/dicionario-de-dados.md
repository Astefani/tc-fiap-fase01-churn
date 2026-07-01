# Dicionário de Dados

**Dataset:** Telco Customer Churn (IBM)  
**Registros:** 7.043 clientes · **Colunas originais:** 50  
**Target:** `ChurnLabel` (Yes/No → 1/0) · **Prevalência de churn:** ~26,5%

Descrições originais do IBM. A coluna **Status** indica o uso no modelo:
✅ mantida · ❌ removida · 🎯 target — justificativas detalhadas em [model-card-aux.md](model-card-aux.md).

---

## Identificação

| Feature | Descrição | Status |
|---|---|---|
| CustomerID | Identificador único do cliente. | ❌ Identificador |

---

## Dados pessoais

| Feature | Descrição | Status |
|---|---|---|
| Gender | Gênero do cliente: Male, Female. | ✅ |
| Age | Idade atual do cliente (anos), ao fim do trimestre fiscal. | ✅ |
| Senior Citizen | Indica se o cliente tem 65 anos ou mais: Yes, No. | ❌ Derivada de Age |
| Under30 | Indica se o cliente tem menos de 30 anos: Yes, No. | ❌ Derivada de Age |
| Married | Indica se o cliente é casado: Yes, No. | ✅ |
| Dependents | Indica se o cliente mora com dependentes: Yes, No. | ❌ Derivada de NumberofDependents |
| Number of Dependents | Número de dependentes que moram com o cliente. | ✅ |

---

## Localização

| Feature | Descrição | Status |
|---|---|---|
| Country | País de residência principal do cliente. | ❌ Valor constante |
| State | Estado de residência principal do cliente. | ❌ Valor constante |
| City | Cidade de residência principal do cliente. | ❌ Alta cardinalidade / redundante com lat/lon |
| Zip Code | CEP da residência principal do cliente. | ❌ Alta cardinalidade / redundante com lat/lon |
| Latitude | Latitude da residência principal do cliente. | ✅ |
| Longitude | Longitude da residência principal do cliente. | ✅ |
| Population | Estimativa populacional atual da área do CEP. | ✅ |

---

## Relacionamento com a empresa

| Feature | Descrição | Status |
|---|---|---|
| Quarter | Trimestre fiscal de referência dos dados (ex.: Q3). | ❌ Valor constante |
| Referred a Friend | Indica se o cliente já indicou um amigo ou familiar: Yes, No. | ❌ Derivada de Number of Referrals |
| Number of Referrals | Número total de indicações feitas pelo cliente até o momento. | ✅ |
| Tenure in Months | Total de meses como cliente da empresa até o fim do trimestre. | ✅ |
| Offer | Última oferta de marketing aceita pelo cliente: None, Offer A–E. | ✅ |

---

## Serviços de telefonia

| Feature | Descrição | Status |
|---|---|---|
| Phone Service | Indica se o cliente assina serviço de telefone fixo: Yes, No. | ✅ |
| Multiple Lines | Indica se o cliente assina múltiplas linhas telefônicas: Yes, No. | ✅ |
| Avg Monthly Long Distance Charges | Média mensal de cobranças de longa distância, calculada até o fim do trimestre. | ✅ |

---

## Serviços de internet

| Feature | Descrição | Status |
|---|---|---|
| Internet Service | Indica se o cliente assina internet e o tipo: No, DSL, Fiber Optic, Cable. | ❌ Substituída por InternetType |
| Internet Type | Tecnologia de internet contratada: DSL, Fiber Optic, Cable, None. | ✅ |
| Avg Monthly GB Download | Média mensal de download em GB, calculada até o fim do trimestre. | ✅ |
| Online Security | Assina serviço adicional de segurança online: Yes, No. | ✅ |
| Online Backup | Assina serviço adicional de backup online: Yes, No. | ✅ |
| Device Protection Plan | Assina plano adicional de proteção de dispositivos: Yes, No. | ✅ |
| Premium Tech Support | Assina suporte técnico premium (menor tempo de espera): Yes, No. | ✅ |
| Streaming TV | Usa a internet para streaming de TV por provedores terceiros: Yes, No. | ✅ |
| Streaming Movies | Usa a internet para streaming de filmes por provedores terceiros: Yes, No. | ✅ |
| Streaming Music | Usa a internet para streaming de música por provedores terceiros: Yes, No. | ✅ |
| Unlimited Data | Pagou mensalidade adicional para downloads/uploads ilimitados: Yes, No. | ✅ |

---

## Contrato e pagamento

| Feature | Descrição | Status |
|---|---|---|
| Contract | Tipo de contrato atual: Month-to-Month, One Year, Two Year. | ✅ |
| Paperless Billing | Optou por faturamento sem papel: Yes, No. | ✅ |
| Payment Method | Forma de pagamento: Bank Withdrawal, Credit Card, Mailed Check. | ✅ |

---

## Cobranças

| Feature | Descrição | Status |
|---|---|---|
| Monthly Charge | Cobrança mensal total atual pelo conjunto de serviços. | ✅ |
| Total Charges | Total cobrado do cliente, calculado até o fim do trimestre. | ✅ |
| Total Refunds | Total de reembolsos recebidos, calculado até o fim do trimestre. | ✅ |
| Total Extra Data Charges | Total de cobranças extras por download acima do plano. | ✅ |
| Total Long Distance Charges | Total de cobranças de longa distância acima do plano. | ✅ |
| Total Revenue | Soma de TotalCharges, TotalLongDistanceCharges, TotalExtraDataCharges e TotalRefunds. | ❌ Agregação redundante |

---

## Target e leakage

| Feature | Descrição | Status |
|---|---|---|
| Churn Label | **Target:** Yes = cliente cancelou no trimestre. No = permaneceu. | 🎯 |
| Customer Status | Status do cliente ao fim do trimestre: Churned, Stayed, Joined. | ❌ Leakage (o próprio desfecho) |
| Satisfaction Score | Nota de satisfação geral do cliente: 1 (muito insatisfeito) a 5 (muito satisfeito). | ❌ Leakage confirmado (PR-AUC 0,79 → 0,98 com a variável) |
| Churn Score | Pontuação de propensão ao churn calculada pelo IBM SPSS Modeler (0–100). | ❌ Leakage |
| CLTV | Customer Lifetime Value predito por fórmulas corporativas. | ❌ Precaução (possível leakage) |
| Churn Category | Categoria do motivo de cancelamento: Attitude, Competitor, Dissatisfaction, Other, Price. | ❌ Leakage (disponível só após o churn) |
| Churn Reason | Motivo específico do cancelamento informado pelo cliente. | ❌ Leakage (disponível só após o churn) |
