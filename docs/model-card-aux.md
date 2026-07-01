# Seleção e Exclusão de Features

**Autor:** Alessandro Stefani  
**Disciplina:** Tech Challenge — Fase 01 (Produtização de Modelos)  
**Dataset:** Telco Customer Churn (IBM)

Documento de apoio ao [Model Card](model-card.md). Registra, para cada feature, a decisão
(manter ou remover) e a justificativa técnica correspondente. As decisões derivam da análise
exploratória (Notebook 01) e, quando aplicável, foram validadas quantitativamente nos baselines.

---

## Features removidas

### Identificador

**CustomerID**  
Removida por representar a identificação única do cliente, sem valor preditivo.

---

### Data leakage

Variáveis que carregam informação do próprio evento de churn — disponíveis apenas após o
cancelamento (ou derivadas dele) e, portanto, inexistentes no momento da predição em produção.

**CustomerStatus**  
Removida por representar o status do cliente, indicando diretamente se houve churn.

**ChurnScore**  
Removida por representar uma pontuação de propensão ao churn já calculada. Sua utilização
introduziria vazamento de informação, uma vez que a pontuação pode ter sido gerada a partir
de variáveis altamente correlacionadas com o evento de cancelamento.

**ChurnCategory**  
Removida por conter uma categorização associada ao motivo do churn. Esta informação só está
disponível após a ocorrência do cancelamento e, portanto, não estaria disponível em um cenário
real de predição.

**ChurnReason**  
Removida por registrar o motivo do cancelamento do cliente — informação conhecida apenas após
a ocorrência do churn, configurando vazamento de informação para o treinamento.

**SatisfactionScore**  
Inicialmente classificada como leakage *suspeito*: a documentação do dataset não especifica o
momento da coleta da pesquisa de satisfação, deixando dúvida sobre sua disponibilidade no
instante da predição. A hipótese foi testada de forma controlada no baseline (Regressão
Logística), comparando o mesmo modelo com e sem a variável, alterando apenas sua presença:

- **Sem** SatisfactionScore: PR-AUC de **0,786**
- **Com** SatisfactionScore: PR-AUC de **0,982**

O salto de 0,786 → 0,982 é desproporcional e incompatível com uma feature legítima, confirmando
que a variável carrega informação do próprio evento de churn (coletada após o cancelamento ou
dele derivada). **Decisão: removida em definitivo.**

---

### Precaução metodológica

**CLTV**  
Removida por corresponder a um Customer Lifetime Value predito. A documentação informa que o
valor é calculado por fórmulas corporativas e dados históricos, sem detalhar as variáveis
utilizadas. Como não foi possível garantir a ausência de informações futuras ou derivadas do
comportamento de churn, a variável foi excluída por precaução metodológica.

---

### Redundância

**ZipCode**  
Removida por representar a mesma informação geográfica já disponível em Latitude e Longitude.
A variável apresenta alta cardinalidade (1.626 valores distintos), o que exigiria estratégias
adicionais de codificação sem ganho informacional evidente sobre as coordenadas. Latitude e
Longitude foram mantidas por preservarem a informação espacial em formato numérico contínuo,
mais adequado aos algoritmos avaliados.

**SeniorCitizen e Under30**  
Removidas por serem transformações determinísticas da variável Age. Como a idade já está
disponível em formato contínuo, manteve-se apenas a variável original, evitando redundância e
preservando o maior nível de informação possível.

**TotalRevenue**  
Removida por corresponder a uma agregação determinística de TotalCharges,
TotalLongDistanceCharges, TotalExtraDataCharges e TotalRefunds. A validação demonstrou
equivalência para todos os registros do dataset. As variáveis componentes foram mantidas por
preservarem maior granularidade e potencial informacional.

**InternetService**  
Removida por indicar apenas a existência do serviço de internet (Sim/Não), enquanto
InternetType fornece maior detalhamento sobre a tecnologia utilizada (DSL, Fiber Optic ou
Cable). Manteve-se a variável mais informativa.

**City**  
Removida por alta cardinalidade e por ser redundante com a informação geográfica já capturada
por Latitude e Longitude.

**ReferredaFriend**  
Removida por ser transformação determinística da variável Number_of_Referrals
(`ReferredaFriend = "Yes"` equivale a `Number_of_Referrals > 0`). Como a quantidade de
indicações já está disponível em formato contínuo, manteve-se apenas a variável original,
evitando redundância e preservando o maior nível de informação possível. A presença das duas
no baseline gerou coeficientes colineares (sinais opostos e grandes), confirmando a redundância.

**Dependents**  
Removida por ser transformação determinística da variável NumberofDependents
(`Dependents = "Yes"` equivale a `NumberofDependents > 0`). Como a quantidade de dependentes
já está disponível em formato contínuo, manteve-se apenas a variável original, evitando
redundância e preservando o maior nível de informação possível.

---

### Valor constante

**Country, State, Quarter**  
Apresentam valor constante em todos os registros (variância zero), não contribuindo para a
discriminação do modelo. Em produção, monitorar se passam a apresentar variação.

---

## Features mantidas após análise

**Latitude e Longitude**  
Mantidas por representarem a localização geográfica do cliente em formato numérico contínuo.
Podem capturar diferenças regionais de comportamento sem os problemas de alta cardinalidade
associados ao código postal.

**Population**  
Mantida por representar características demográficas da região associada ao cliente. Embora não
descreva diretamente o indivíduo, pode atuar como variável contextual relacionada ao ambiente
competitivo, à densidade populacional e ao perfil socioeconômico da localidade.

**InternetType**  
Mantida em lugar de InternetService por detalhar a tecnologia de internet do cliente. Os nulos
correspondem a clientes sem serviço de internet (confirmado por cruzamento com InternetService)
e foram imputados como "No internet".

**Offer**  
Mantida. Os valores ausentes foram interpretados como ausência de oferta aceita pelo cliente —
parte do significado de negócio da variável, e não falha de preenchimento. Imputados como
"No offer".

---

## Quadro-resumo

| Feature           | Decisão  | Motivo                                        |
|-------------------|----------|-----------------------------------------------|
| CustomerID        | Removida | Identificador                                 |
| CustomerStatus    | Removida | Data leakage                                  |
| ChurnScore        | Removida | Data leakage                                  |
| ChurnCategory     | Removida | Data leakage                                  |
| ChurnReason       | Removida | Data leakage                                  |
| SatisfactionScore | Removida | Data leakage confirmado (PR-AUC 0,79 → 0,98) |
| CLTV              | Removida | Precaução (possível leakage)                  |
| ZipCode           | Removida | Redundância geográfica + alta cardinalidade   |
| SeniorCitizen     | Removida | Redundância (derivada de Age)                 |
| Under30           | Removida | Redundância (derivada de Age)                 |
| TotalRevenue      | Removida | Redundância (agregação determinística)        |
| InternetService   | Removida | Redundância (InternetType é mais detalhada)   |
| City              | Removida | Alta cardinalidade / geo redundante           |
| ReferredaFriend   | Removida | Redundância (derivada de Number_of_Referrals) |
| Dependents        | Removida | Redundância (derivada de NumberofDependents)  |
| Country           | Removida | Valor constante                               |
| State             | Removida | Valor constante                               |
| Quarter           | Removida | Valor constante                               |
| Latitude          | Mantida  | Informação geográfica                         |
| Longitude         | Mantida  | Informação geográfica                         |
| Population        | Mantida  | Contexto regional                             |
| InternetType      | Mantida  | Detalha a tecnologia de internet              |
| Offer             | Mantida  | Missing com significado de negócio            |
