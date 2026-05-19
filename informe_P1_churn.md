# Análisis de Retención: Predicción de Churn en Telecomunicaciones

**Proyecto 1 — Clasificación | Machine Learning (CS3061)**

**Autores:** [Nombre 1] · [Nombre 2] · [Nombre 3] · [Nombre 4]

**Profesor:** Prof. D.Sc. Manuel Eduardo Loaiza Fernandez

---

## Resumen Ejecutivo

Se desarrolló un sistema de clasificación binaria para predecir la cancelación (*churn*) de clientes de una empresa de telecomunicaciones. El conjunto de datos contiene 5,634 registros con desbalance notable (73.4% retenidos vs. 26.6% fugados), variables mixtas y correlacionadas. Se implementaron Regresión Logística y Random Forest, optimizados mediante validación cruzada estratificada de 5 pliegues con SMOTE. Dado que el costo de un Falso Negativo ($200) supera diez veces al de un Falso Positivo ($20), se ajustó el umbral de decisión minimizando el costo total esperado. La Regresión Logística con τ\*=0.16 alcanza el menor costo esperado ($11,200) y el mayor AUC (0.852), siendo el modelo recomendado. El tipo de contrato, el cargo mensual y la antigüedad son los predictores de mayor peso en la fuga de clientes.

---

## 1. Introducción

La retención de clientes es uno de los desafíos principales en telecomunicaciones, donde adquirir un nuevo cliente cuesta entre cinco y diez veces más que retener uno existente. El *churn* o abandono genera pérdidas directas de ingresos y deteriora indicadores clave como el valor de vida del cliente (LTV).

Este proyecto aborda el problema como una tarea de **clasificación binaria supervisada**: dado el perfil de consumo de un cliente, predecir si cancelará el servicio. Se trabaja con el dataset *Telco Customer Churn*, con 7,043 registros divididos en entrenamiento (5,634) y prueba (1,411).

En la literatura, los modelos de clasificación para churn típicamente combinan análisis de comportamiento (uso de servicios, historial de pagos) con características sociodemográficas del cliente. La clave del éxito no radica únicamente en el algoritmo seleccionado, sino en la correcta formulación del problema: la elección de la métrica de optimización, el manejo del desbalance y la calibración del umbral de decisión según el contexto de negocio.

**Desafíos principales:**

**(1) Desbalance de clases:** solo el 26.6% de los clientes presenta churn, lo que penaliza la detección de la clase minoritaria sin estrategias de compensación. **(2) Datos mixtos:** variables numéricas continuas, categóricas de alta cardinalidad y binarias requieren pipelines de preprocesamiento diferenciados. **(3) Asimetría de costos:** perder un cliente tiene impacto económico diez veces superior a una campaña de retención innecesaria, exigiendo calibración explícita del umbral de decisión.

---

## 2. Ingeniería de Datos y Formulación

### 2.1 Descripción del Dataset

El dataset cuenta con **20 variables predictoras** y la variable objetivo Churn. La Tabla 1 resume las características principales de cada variable.

| Variable | Tipo | Descripción y estadística |
|---|---|---|
| gender | Binaria | Género (51.2% M / 48.8% F) |
| SeniorCitizen | Binaria | Mayor de 65 años (16.1%) |
| Partner | Binaria | Tiene pareja (48.5%) |
| Dependents | Binaria | Tiene dependientes (29.9%) |
| tenure | Numérica | Meses como cliente (0–72, x̄=32.4) |
| PhoneService | Binaria | Servicio telefónico (90.2%) |
| InternetService | Categ. | DSL (34.4%) / Fibra (44%) / No (21.5%) |
| Contract | Categ. | Mensual (54.7%) / Anual / Bianual |
| PaymentMethod | Categ. | 4 métodos de pago |
| MonthlyCharges | Numérica | Cargo mensual ($18–$119, x̄=$64.9) |
| TotalCharges | Numérica | Cargo acumulado ($19–$8,685) |
| Churn | Objetivo | Canceló el servicio (26.6%) |

*Tabla 1: Variables del dataset Telco Customer Churn.*

### 2.2 Análisis Exploratorio de Datos

**Desbalance de clases.** 4,138 clientes retenidos (73.4%) y 1,496 fugados (26.6%). Este desbalance exige estrategias de compensación para no ignorar la clase minoritaria.

**Antigüedad y churn.** Clientes con churn se concentran en los primeros 10 meses (media ≈18 meses vs. ≈38 meses en retenidos). La fase de incorporación es el período crítico de riesgo.

**Tipo de contrato.** Mayor predictor discriminante: contratos mensuales con 42.7% de churn, frente a 11.8% en anuales y solo 2.8% en bianuales — diferencia de 15×.

**Otros patrones.** Fibra óptica: 41.6% churn vs. 19.2% DSL. Pago por cheque electrónico: 45.0% churn (el más alto). Adultos mayores: 41.4% churn vs. 23.7% en no mayores.

**Análisis de cargos.** El cargo mensual promedio de los clientes que cancelan es $74.5 frente a $61.3 de los que permanecen — diferencia del 21.5%. Los servicios de streaming (TV y películas) están presentes en el 44% de los clientes con churn, sugiriendo que este segmento busca alternativas más económicas. Los servicios de seguridad en línea y soporte técnico, en cambio, se asocian con menor churn (25.5% vs. 43.2% en quienes no tienen soporte técnico), indicando que la asistencia al cliente actúa como factor de retención.

> *Figura 1: Análisis exploratorio. Izq.: desbalance de clases (26.6% churn). Centro: distribución de tenure por clase. Der.: tasa de churn por tipo de contrato.*

### 2.3 Variable Objetivo

y_i = 1 si el cliente i canceló (Churn = Yes), 0 si permaneció activo.

### 2.4 Nuevas Variables Creadas

Se construyeron seis variables mediante ingeniería de características para capturar señales no presentes directamente en los datos originales (Tabla 2).

| Variable | Fórmula / Justificación |
|---|---|
| num_services | Σ servicios activos (0–6). Mayor dependencia → menor churn. |
| avg_monthly_spend | TotalCharges / max(tenure, 1). Gasto promedio real. |
| is_new_customer | 1 si tenure ≤ 3 meses. Período crítico de abandono temprano. |
| has_protection | 1 si tiene DeviceProtection o TechSupport. Proxy de compromiso tecnológico. |
| is_fiber_optic | 1 si InternetService = Fiber optic. Mayor tasa de churn observada (41.6%). |
| charge_per_service | MonthlyCharges / max(num_services, 1). Detecta clientes que pagan mucho por poco. |

*Tabla 2: Variables de ingeniería de características.*

### 2.5 Pipeline de Preprocesamiento

**Valores faltantes.** TotalCharges presenta nulos en 11 registros con tenure=0 (clientes sin período de facturación). Se imputan con MonthlyCharges.

**Codificación.** Variables binarias Yes/No → 0/1. Las etiquetas "No phone service" y "No internet service" se homologan a "No" (0). InternetService, Contract y PaymentMethod se codifican con *one-hot encoding*, expandiendo el espacio de 20 a **32 features**.

**Escalado.** Las variables numéricas continuas se estandarizan con StandardScaler (μ=0, σ=1), calculado únicamente sobre datos de entrenamiento para evitar fuga de información al conjunto de validación.

**Partición del dataset.** El dataset se divide en entrenamiento (80%, 4,507 muestras) y holdout de validación (20%, 1,127 muestras) usando partición estratificada que preserva la proporción de clases en ambos conjuntos. Esta separación garantiza que la evaluación final sea sobre datos completamente no vistos durante el entrenamiento y la selección de hiperparámetros.

**Desbalance de clases.** Se aplica SMOTE (*Synthetic Minority Oversampling Technique*) exclusivamente dentro de cada pliegue de entrenamiento en la validación cruzada, evitando la contaminación del pliegue de validación con datos sintéticos — error que inflaría artificialmente las métricas. Se emplea también class_weight="balanced" en ambos modelos. Tras SMOTE, el conjunto de entrenamiento queda balanceado al 50%–50%.

### 2.6 Análisis de Correlaciones

Se calculó la matriz de correlación de Pearson sobre las variables numéricas tras preprocesamiento. TotalCharges y tenure presentan la correlación más alta (ρ ≈ 0.83), lo que tiene una explicación directa: a mayor tiempo como cliente, mayor es el cargo acumulado. Para evitar colinealidad, la feature de ingeniería avg_monthly_spend (TotalCharges / tenure) captura esta relación como una variable normalizada que mide el comportamiento de gasto real del cliente.

MonthlyCharges y TotalCharges también presentan correlación moderada (ρ ≈ 0.65). La regularización ℓ₁ en la Regresión Logística maneja esta colinealidad automáticamente al suprimir uno de los predictores redundantes. En el Random Forest, la aleatorización de features por nodo (max_features=sqrt) desensibiliza el ensamble ante correlaciones entre predictores.

Las variables de ingeniería charge_per_service y avg_monthly_spend presentan correlaciones bajas con las variables originales (ρ < 0.40), lo que confirma que aportan información nueva al modelo. num_services presenta correlación negativa con churn (ρ ≈ −0.30), validando la hipótesis de que clientes más integrados tienen menor probabilidad de abandono.

---

## 3. Metodología y Optimización

### 3.1 Justificación de Modelos

**Regresión Logística (RL).** Modela la probabilidad de churn mediante:

> P(y=1 | x) = σ(w^T x + b) = 1 / (1 + e^{−(w·x+b)})

Ventajas en este contexto: (1) salida probabilística calibrada para ajustar el umbral; (2) regularización ℓ₁ realiza selección automática de variables, controlando la colinealidad entre servicios contratados; (3) coeficientes interpretables en términos de log-odds, facilitando la comunicación con el equipo de negocio.

**Random Forest (RF).** Ensamble de B árboles de decisión entrenados sobre muestras bootstrap con subconjunto aleatorio de features por división. Ventajas: (1) captura relaciones no lineales e interacciones entre variables — contrato mensual + fibra óptica + sin seguridad → churn > 50%; (2) el promedio de múltiples árboles reduce la varianza respecto a un árbol individual; (3) importancia de Gini como medida nativa de relevancia de variables.

### 3.2 Esquema de Validación

Se reserva un **holdout estratificado del 20%** (1,127 muestras) para evaluación final, preservando la proporción de clases. El ajuste de hiperparámetros se realiza sobre el 80% restante mediante GridSearchCV (RL: 10 combinaciones × 5 pliegues = 50 fits) y RandomizedSearchCV con 25 iteraciones (RF: 125 fits). Métrica de optimización: **F1-Score** sobre la clase positiva (Churn=1).

### 3.3 Hiperparámetros Explorados y Valores Óptimos

| Modelo | Hiperparámetro | Rango | Óptimo |
|---|---|---|---|
| Reg. Logística | C (inv. reg.) | {0.01, 0.1, 1, 10, 100} | 10 |
| | Penalización | {ℓ₁, ℓ₂} | ℓ₁ |
| | Mejor CV F1 | — | 0.8305 |
| Random Forest | n_estimators | [100, 400] | 100 |
| | max_depth | {None, 5, 10, 15, 20} | None |
| | min_samples_split | [2, 15] | 2 |
| | max_features | {sqrt, log2} | sqrt |
| | Mejor CV F1 | — | 0.8486 |

*Tabla 3: Hiperparámetros, rangos explorados y valores óptimos (CV estratificado 5 pliegues, métrica: F1-Score).*

**Consideraciones de complejidad.** La Regresión Logística con liblinear entrena en tiempo O(n·d) por iteración, siendo altamente eficiente para el dataset de 32 features. El Random Forest con 100 árboles y profundidad sin restricción tiene mayor costo computacional O(n·d·log(n)·B), justificando el uso de RandomizedSearchCV (25 iteraciones) en lugar de búsqueda exhaustiva. El tiempo total de entrenamiento con búsqueda de hiperparámetros fue de aproximadamente 2-3 minutos en un entorno estándar con paralelismo (n_jobs=-1).

El valor C=10 (regularización débil) indica que las 32 features son en su mayoría relevantes. La penalización ℓ₁ selecciona automáticamente las variables informativas. Para el RF, max_depth=None prioriza bajo sesgo; la varianza se controla mediante el promedio de 100 árboles independientes.

### 3.4 Configuración del Pipeline Completo

El pipeline de entrenamiento encapsula las siguientes etapas en orden: (1) Imputación de valores faltantes en TotalCharges, (2) Codificación de variables binarias y one-hot encoding de categóricas, (3) Aplicación de SMOTE dentro de cada pliegue de CV para balancear la clase minoritaria, (4) Estandarización mediante StandardScaler ajustado en el pliegue de entrenamiento, (5) Entrenamiento del clasificador con los hiperparámetros propuestos.

La encapsulación del preprocesamiento dentro del CV es fundamental para evitar *data leakage*. Un error habitual es aplicar SMOTE antes de la partición en pliegues: esto introduce muestras sintéticas derivadas del pliegue de validación en el conjunto de entrenamiento, generando estimaciones optimistas imposibles de replicar en producción. El uso de imbalanced-learn Pipeline con SMOTE integrado dentro del ciclo de búsqueda de hiperparámetros garantiza la correcta separación.

La elección de F1-Score como métrica de optimización en la búsqueda de hiperparámetros es deliberada: refleja el balance entre precisión y recall en la clase positiva. Para el contexto del negocio, se prioriza el recall (identificar clientes que van a fugarse) sobre la precisión, lo que se gestiona posteriormente mediante el ajuste del umbral de decisión basado en la matriz de costos.

> *Figura 2: Izq.: costo esperado vs. umbral — RL óptimo τ\*=0.16 ($11,200), RF τ\*=0.08 ($12,080). Der.: curvas ROC — RL AUC=0.852, RF AUC=0.821.*

---

## 4. Análisis de Costo-Beneficio

### 4.1 Matriz de Costos Monetaria

Los dos tipos de error tienen consecuencias económicas asimétricas que deben cuantificarse para orientar la optimización del modelo:

| | Pred.: Retenido (0) | Pred.: Churn (1) |
|---|---|---|
| **Real: Retenido (0)** | $0 — TN | $20 — FP |
| **Real: Churn (1)** | $200 — FN | $0 — TP |

*Tabla 4: Matriz de costos monetaria (USD).*

**Falso Negativo (FN) — $200:** cliente perdido sin intervención. Incluye LTV perdido y costo de adquisición de cliente de reposición.

**Falso Positivo (FP) — $20:** campaña de retención innecesaria. Costo del incentivo ofrecido (descuento o mejora de plan).

La relación C_FN/C_FP = 10:1 justifica formalmente ajustar el umbral por debajo de 0.5: es preferible generar más falsas alarmas que perder clientes reales.

### 4.2 Costo Total Esperado y Ajuste del Umbral

> C_total(τ) = C_FN · |FN(τ)| + C_FP · |FP(τ)|

Se evalúa esta ecuación para τ ∈ [0.05, 0.90] (Figura 2, panel izquierdo). La Tabla 5 cuantifica el impacto económico sobre el holdout (1,127 muestras, 299 clientes con churn real):

| Modelo / τ | FN | FP | Costo ($) | Ahorro |
|---|---|---|---|---|
| RL — τ=0.50 | 89 | 135 | 20,500 | — |
| RL — τ\*=0.16 | 17 | 390 | 11,200 | 45% |
| RF — τ=0.50 | 121 | 139 | 26,980 | — |
| RF — τ\*=0.08 | 5 | 554 | 12,080 | 55% |

*Tabla 5: Comparación de costo total con umbral por defecto vs. umbral óptimo.*

El ajuste del umbral de la RL reduce los clientes perdidos sin intervención de 89 a solo 17 (reducción del 81%), manteniendo el costo en $11,200. El RF con τ\*=0.08 reduce los FN a 5 pero genera 554 FP, resultando en costo ligeramente superior ($12,080). El modelo recomendado para producción es la **Regresión Logística con τ\*=0.16**.

> *Figura 3: Matrices de confusión con umbral óptimo. RL (τ\*=0.16): TP=282, FP=390, FN=17, TN=438. RF (τ\*=0.08): TP=294, FP=554, FN=5, TN=274.*

---

## 5. Evaluación de Resultados

### 5.1 Métricas Comparativas

| Modelo / Umbral | Acc. | F1 | AUC | Costo |
|---|---|---|---|---|
| RL — τ=0.50 | 0.801 | 0.652 | 0.852 | $20,500 |
| RL — τ\*=0.16 | 0.639 | 0.581 | 0.852 | $11,200 |
| RF — τ=0.50 | 0.769 | 0.578 | 0.821 | $26,980 |
| RF — τ\*=0.08 | 0.504 | 0.513 | 0.821 | $12,080 |

*Tabla 6: Métricas en el holdout de validación (1,127 muestras, 20% estratificado).*

Con umbral por defecto, la RL supera al RF en F1 (0.652 vs. 0.578) y AUC (0.852 vs. 0.821). El AUC mide la capacidad discriminante en todo el rango de umbrales: la ventaja de la RL indica mejor calibración probabilística global. Al optimizar el umbral, ambos modelos sacrifican precisión para ganar exhaustividad en la clase churn, reduciendo fuertemente el costo esperado.

> *Figura 4: Izq.: importancia Gini (RF) — charge_per_service lidera, validando el valor de las features de ingeniería. Der.: coeficientes RL — Contract_Month-to-month (β=−4.99) y MonthlyCharges (β=2.66) con mayor magnitud.*

### 5.2 Análisis de Importancia de Variables

La Figura 4 muestra las 15 variables de mayor importancia para ambos modelos. **charge_per_service** (MonthlyCharges / num_services) lidera en el RF, validando el valor de las features de ingeniería. Su alta importancia indica que el modelo captura una señal potente: clientes que pagan mucho pero usan pocos servicios tienen mayor incentivo para cancelar o migrar a un competidor.

**En la Regresión Logística**, el coeficiente de mayor magnitud corresponde a Contract_Month-to-month (β = −4.99), seguido de MonthlyCharges (β = 2.66) y tenure (β = −2.13). El signo negativo de Contract_Month-to-month indica que los contratos anuales/bianuales reducen fuertemente el churn. El signo negativo de tenure confirma que la fidelización actúa como factor protector contra la cancelación.

**Convergencia entre modelos.** Ambos modelos coinciden en identificar los mismos 5 predictores principales: tipo de contrato, cargo mensual, antigüedad, servicio de internet de fibra óptica y método de pago electrónico. Esta consistencia refuerza la robustez de los hallazgos: no son artefactos de un algoritmo particular, sino patrones reales en los datos.

### 5.3 Análisis del Sesgo-Varianza

**Regresión Logística.** F1 en CV = 0.831 vs. F1 en holdout = 0.652 (τ=0.5). La brecha es esperable: el CV opera con datos SMOTE balanceados mientras el holdout tiene la distribución real. La regularización ℓ₁ con C=10 introduce sesgo leve controlando la varianza. El modelo presenta **varianza controlada** con posible sesgo residual al no capturar interacciones no lineales.

**Random Forest.** F1 en CV = 0.849 pero AUC en holdout = 0.821 < 0.852 de la RL. El RF con max_depth=None memoriza patrones del entrenamiento que no generalizan bien: signo de **varianza ligeramente elevada** (sobreajuste leve). Restringir max_depth o aumentar min_samples_split reduciría la varianza a costo de mayor sesgo, mejorando potencialmente la generalización.

---

## 6. Regresión Logística: Análisis Univariado y Multivariado

### 6.1 Feature Engineering para el Análisis

Se seleccionan tenure y MonthlyCharges — los dos predictores numéricos de mayor poder discriminante — estandarizados individualmente (μ=0, σ=1) para facilitar la comparación de coeficientes y la visualización de los decision boundaries.

### 6.2 Caso 1 — Univariado: tenure

> P(y=1 | x_ten) = σ(β₀ + β₁·x_ten)
>
> β̂₀ = −0.1922,  β̂₁ = −0.9081,  CV-F1 = 0.514

El coeficiente negativo β̂₁ < 0 cuantifica la relación inversa entre antigüedad y probabilidad de churn: por cada desviación estándar de incremento en tenure (≈24 meses), los odds de churn se multiplican por e^(−0.908) ≈ 0.40 — reducción del 60%. El decision boundary univariado (P=0.5) se ubica en x\* = −β̂₀/β̂₁ = −0.212 (estandarizado), equivalente a ≈27 meses de antigüedad.

### 6.3 Caso 2 — Univariado: MonthlyCharges

> P(y=1 | x_mon) = σ(β₀ + β₁·x_mon)
>
> β̂₀ = −0.0509,  β̂₁ = 0.4782,  CV-F1 = 0.471

El coeficiente positivo β̂₁ > 0 confirma que cargos más altos incrementan la probabilidad de churn: por cada σ de incremento (≈$30), los odds se multiplican por e^(0.478) ≈ 1.61. El decision boundary se ubica en x\* = 0.107 (estandarizado), equivalente a ≈$68/mes — cercano a la media del dataset ($64.9).

### 6.4 Caso 3 — Modelo Multivariable

> P(y=1 | x) = σ(β₀ + β₁·x_ten + β₂·x_mon)
>
> β̂₀=−0.394,  β̂₁=−1.286,  β̂₂=0.964,  CV-F1=0.573

Al combinar ambos predictores, los coeficientes aumentan en magnitud (|β̂₁|: 0.908→1.286; |β̂₂|: 0.478→0.964), indicando que tenure y MonthlyCharges proveen información **complementaria, no redundante**. El decision boundary multivariable es el hiperplano lineal β₀ + β₁x₁ + β₂x₂ = 0, visualizable en 2D como la recta: x₂ = (0.394 + 1.286·x₁) / 0.964. Clientes con alta carga y baja antigüedad quedan en la región de mayor riesgo.

### 6.5 Comparación de Contribuciones Probabilísticas

La mejora al pasar del modelo univariado de tenure (CV-F1=0.514) al multivariable (CV-F1=0.573) es de +11.5% en F1, confirmando que precio y lealtad son señales independientes. Sin embargo, ninguno de estos modelos simples se acerca al rendimiento del modelo completo de 32 features (CV-F1≈0.831), lo que ilustra que los servicios contratados, el tipo de contrato y el método de pago aportan información predictiva adicional no capturada por las dos variables numéricas. Esto valida el enfoque de ingeniería de características adoptado en este proyecto.

- **Caso 1 (tenure solo):** CV-F1 = 0.514 — captura fidelización, ignora precio.
- **Caso 2 (MonthlyCharges solo):** CV-F1 = 0.471 — captura precio, ignora lealtad.
- **Caso 3 (multivariable):** CV-F1 = 0.573 — mejora del **+11.5%** sobre el mejor univariado.

La mejora adicional al usar las 32 features del modelo completo (CV-F1 ≈ 0.831) ilustra cuantitativamente el valor de las variables de ingeniería de características.

---

## 7. Conclusiones

**Hallazgo principal.** La Regresión Logística con τ\*=0.16 es el modelo recomendado para producción: alcanza el menor costo total esperado ($11,200), una reducción del 45% respecto al umbral por defecto ($20,500), y el mejor AUC global (0.852). Con este umbral, de 299 clientes que cancelan, el modelo identifica correctamente 282 (recall del 94.3%), generando 390 falsas alarmas a $20 c/u.

**Análisis de precision y recall.** Con τ\*=0.16, la RL presenta un recall del 94.3% sobre la clase churn (282 de 299 clientes identificados correctamente), a costo de una precisión reducida del 42.0% (282 TP de 672 alertas totales). Este tradeoff es **deliberado y justificado por la matriz de costos**: cada TP evita una pérdida de $200, y cada FP cuesta solo $20 en campaña innecesaria.

**Limitaciones.** La evaluación se realizó en un holdout interno; la métrica real debe medirse en un A/B test con el sistema de retención activo. La matriz de costos ($200 FN, $20 FP) es una estimación que debe calibrarse con datos reales de LTV por segmento. SMOTE puede generar muestras poco realistas en presencia de outliers extremos.

**Lecciones aprendidas.** (1) El ajuste del umbral de decisión según la matriz de costos supera en impacto económico a la elección del algoritmo. (2) Aplicar SMOTE dentro del ciclo de CV es crítico para evitar estimaciones optimistas. (3) Las features de ingeniería (charge_per_service, avg_monthly_spend) aparecen en el top-5 del RF, validando su aportación predictiva. (4) La convergencia entre ambos modelos en los mismos predictores clave refuerza la interpretabilidad de los hallazgos.

**Recomendaciones.** Implementar monitoreo de deriva de distribución (*concept drift*) sobre el score de churn en producción. Explorar modelos secuenciales (LSTM) con datos históricos mensuales. Segmentar la matriz de costos por tipo de cliente para umbrales diferenciados.

**Trabajo futuro.** Tres direcciones prometedoras para mejorar el sistema: (1) *Gradient Boosting* (XGBoost, LightGBM) que típicamente supera a RF en datasets tabulares de este tamaño; (2) calibración de probabilidades mediante Platt Scaling o regresión isotónica, para que P̂(y=1|x) refleje frecuencias reales y el umbral τ\* sea más estable; (3) análisis de supervivencia (Cox proportional hazards) que modela el tiempo hasta el churn en lugar de una predicción binaria estática.

---

## 8. Reproducibilidad y Recursos

Todo el código está disponible en el repositorio indicado. Para garantizar reproducibilidad completa, se fijó random_state=42 en todas las operaciones estocásticas: partición train/holdout, StratifiedKFold, SMOTE, RandomizedSearchCV y los clasificadores. El entorno de ejecución es Google Colab (Python 3.10, scikit-learn 1.3, imbalanced-learn 0.11).

**Versiones de dependencias clave:** scikit-learn 1.3.x, imbalanced-learn 0.11.x, pandas 2.0.x, numpy 1.24.x, matplotlib 3.7.x, seaborn 0.12.x. Se incluye una celda de instalación automática en el notebook para garantizar compatibilidad en cualquier entorno Colab nuevo.

**Dataset.** Telco Customer Churn de IBM — disponible públicamente. La división train/test provista en el curso (5,634 / 1,411 filas) se respeta sin modificación para comparabilidad entre equipos. Los archivos de predicción sobre el conjunto de prueba se exportan en formato CSV con columna única "Churn" (0/1) con el umbral óptimo τ\*=0.16 de la RL.

**Notebook.** El archivo P1_Churn_Telco_Clasificacion.ipynb contiene 39 celdas estructuradas en secciones correspondientes a las del presente informe. La ejecución secuencial completa (Run All) requiere aproximadamente 4-5 minutos en Colab con CPU estándar, incluyendo la búsqueda de hiperparámetros con paralelismo (n_jobs=-1). Las 4 figuras del informe se generan automáticamente y se guardan en disco en formato PNG y PDF de alta resolución (300 DPI).

**Repositorio:** [URL del repositorio GitHub/Colab]

---

## Contribution Statement

| Integrante | Contribución específica | % |
|---|---|---|
| [Nombre 1] | EDA, ingeniería de las 6 features nuevas, análisis exploratorio, redacción Secciones 1–2. | 25% |
| [Nombre 2] | Pipeline de preprocesamiento (codificación, escalado, SMOTE), Regresión Logística con GridSearchCV, Sección 6 (casos 1-3). | 25% |
| [Nombre 3] | Random Forest con RandomizedSearchCV, importancia de variables, curvas de aprendizaje (bias-variance), Secciones 3–4. | 25% |
| [Nombre 4] | Matriz de costos, optimización de umbral, generación de figuras, integración del notebook final, Secciones 5 y 7, revisión general. | 25% |

*Tabla 7: Distribución de contribuciones del equipo.*
