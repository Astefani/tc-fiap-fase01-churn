**Autor:** Alessandro Stefani
**Disciplina:** Tech Challenge — Fase 01 (Produtização de Modelos)
**Dataset:** Telco Customer Churn (IBM)


# Seleção e Exclusão de Features

> Features removidas por serem identificadores

**CustomerID**  
Removida por representar a identificação do cliente no dataset.


> Features removidas por risco de Data Leakage

**CustomerStatus**  
Removida por representar o status do cliente, indicando se houve churn.

**ChurnScore**  
Removida por representar uma pontuação de propensão ao churn já calculada. A utilização desta variável poderia introduzir vazamento de informação (data leakage), uma vez que a pontuação pode ter sido gerada a partir de variáveis altamente correlacionadas com o evento de cancelamento.

**ChurnCategory**  
Removida por conter uma categorização associada ao motivo do churn. Esta informação somente está disponível após a ocorrência do evento de cancelamento e, portanto, não estaria disponível em um cenário real de predição.

**ChurnReason**  
Removida por registrar o motivo do cancelamento do cliente. Trata-se de uma informação conhecida apenas após a ocorrência do churn, configurando vazamento de informação para o treinamento do modelo.

**CLTV**  
Removida por corresponder a um Customer Lifetime Value predito. A documentação do dataset informa que o valor é calculado por meio de fórmulas corporativas e dados históricos, sem detalhar as variáveis utilizadas no cálculo. Como não foi possível garantir a ausência de informações futuras ou derivadas do comportamento de churn, a variável foi excluída por precaução metodológica.


> Features removidas por redundância  

**ZipCode**  
Removida por representar a mesma informação geográfica já disponível nas variáveis Latitude e Longitude. A variável apresenta alta cardinalidade (1.626 valores distintos), o que exigiria estratégias adicionais de codificação sem fornecer ganho informacional evidente em relação às coordenadas geográficas.
As variáveis Latitude e Longitude foram mantidas por preservarem a informação espacial em formato numérico contínuo, mais adequado para utilização pelos algoritmos de aprendizado de máquina avaliados neste projeto.

**SeniorCitizen e Under30**  
Removidas por serem transformações determinísticas da variável Age. Como a idade já está disponível em formato contínuo, optou-se por manter apenas a variável original para evitar redundância e preservar o maior nível possível de informação.

**TotalRevenue**  
A variável TotalRevenue foi removida do conjunto de treinamento por corresponder a uma agregação determinística das variáveis TotalCharges, TotalLongDistanceCharges, TotalExtraDataCharges e TotalRefunds. A validação realizada demonstrou equivalência para todos os registros do dataset. As variáveis componentes foram mantidas por preservarem maior granularidade e potencial informacional para o modelo.

**InternetService**  
Removida após ser observado que a variável InternetService indicava apenas a existência do serviço de internet (Sim/Não), enquanto InternetType fornecia maior nível de detalhamento sobre a tecnologia utilizada pelo cliente (DSL, Fiber Optic ou Cable).

**City**  
Removida devido a alta cardinalidade e ser relacionada com Lat e Long.


> Features removidas por valor constante  

- **Country, State, Quarter**  
Apresentam valores fixos para todas variáveis. Observar se em produção aparecem valores diferentes.



> Features mantidas após análise exploratória  

**Latitude e Longitude**  
Mantidas por representarem a localização geográfica do cliente em formato numérico contínuo. Essas variáveis podem capturar diferenças regionais de comportamento sem introduzir os problemas de alta cardinalidade associados ao código postal.

**Population**  
Mantida por representar características demográficas da região associada ao cliente. Embora não descreva diretamente o indivíduo, pode atuar como variável contextual relacionada ao ambiente competitivo, densidade populacional e perfil socioeconômico da localidade.

**SatisfactionScore**  
Mantida para avaliação experimental. A documentação disponível não especifica o momento exato da coleta da pesquisa de satisfação. Como não foram encontradas evidências de que a informação seja obtida após o cancelamento do cliente, a variável foi mantida nesta versão do modelo. Recomenda-se reavaliar sua utilização caso novas informações sobre sua origem e temporalidade se tornem disponíveis.


**Offer**  
Mantida. Os valores ausentes foram interpretados como ausência de oferta aceita pelo cliente, e não como falha de preenchimento. Dessa forma, os valores nulos foram considerados parte do significado de negócio da variável.



| Feature           | Decisão  | Motivo                             |
| ----------------- | -------- | ---------------------------------- |
| CustomerID        | Removida | Identificador                      |
| ChurnScore        | Removida | Data leakage                       |
| ChurnCategory     | Removida | Data leakage                       |
| ChurnReason       | Removida | Data leakage                       |
| CLTV              | Removida | Possível data leakage              |
| ZipCode           | Removida | Redundância geográfica             |
| Latitude          | Mantida  | Informação geográfica              |
| Longitude         | Mantida  | Informação geográfica              |
| Population        | Mantida  | Contexto regional                  |
| SatisfactionScore | Mantida  | Sem evidência de leakage           |
| Offer             | Mantida  | Missing com significado de negócio |
