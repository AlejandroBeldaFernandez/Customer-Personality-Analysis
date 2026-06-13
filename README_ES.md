# Customer Personality Analysis — Segmentacion de Clientes No Supervisada

Proyecto de machine learning no supervisado para segmentar clientes en funcion de su perfil demografico, comportamiento de compra y respuesta a campanas de marketing, usando un dataset publico de 2.240 clientes.

- **Problema:** Agrupar clientes en segmentos significativos sin etiquetas predefinidas, usando unicamente sus datos transaccionales y demograficos
- **Resultado:** 3 segmentos identificados (Premium, Cazador de Ofertas, Escaparatista) con una silueta de 0.24 tras reduccion de dimensionalidad con PCA
- **Valor:** Permite estrategias de marketing personalizadas por segmento — evitando malgastar presupuesto promocional y aumentando el ROI de las campanas

> [View this project in English](README.md)

---

## Tabla de contenidos

1. [Definicion del problema](#definicion-del-problema)
2. [Valor de negocio](#valor-de-negocio)
3. [Dataset](#dataset)
4. [Retos y transformaciones de los datos](#retos-y-transformaciones-de-los-datos)
5. [Analisis exploratorio de datos](#analisis-exploratorio-de-datos)
6. [Metodologia](#metodologia)
7. [Pipeline de preprocesamiento](#pipeline-de-preprocesamiento)
8. [Seleccion del modelo](#seleccion-del-modelo)
9. [Segmentos de clientes](#segmentos-de-clientes)
10. [Limitaciones del modelo](#limitaciones-del-modelo)
11. [Conclusiones](#conclusiones)
12. [Posibles mejoras](#posibles-mejoras)
13. [Requisitos](#requisitos)

---

## Definicion del problema

Entender a los clientes como individuos en lugar de como un grupo homogeneo es clave para un marketing eficaz. Las empresas que pueden identificar perfiles de cliente diferenciados estan mejor posicionadas para personalizar sus ofertas, asignar presupuestos de campana de forma eficiente y mejorar la retencion.

Este proyecto aborda la siguiente pregunta:

> **Dados datos demograficos, transaccionales y de respuesta a campanas, podemos identificar segmentos de clientes que soporten estrategias de marketing diferenciadas?**

Se plantea como un problema de **clustering no supervisado** — no hay etiquetas predefinidas. El modelo debe descubrir la estructura en los datos desde cero.

---

## Valor de negocio

Identificar segmentos de clientes tiene valor comercial directo:

- **Campanas personalizadas.** Los clientes Premium reciben programas de fidelizacion y acceso exclusivo; los Cazadores de Ofertas reciben promociones dirigidas; los Escaparatistas reciben ofertas familiares en web. Cada segmento recibe el mensaje con mayor probabilidad de conversion.
- **Optimizacion del presupuesto.** Los descuentos dirigidos a clientes Premium que compran a precio completo de todos modos son gasto perdido. La segmentacion permite asignar el presupuesto promocional donde tiene mayor ROI.
- **Foco en la retencion.** Saber a que segmento pertenece un cliente permite disenar estrategias de retencion especificas antes de que se produzca la baja.

---

## Dataset

- **Fuente:** [Customer Personality Analysis — Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis)
- **Registros:** 2.240 clientes antes de la limpieza
- **Variables:** 29 variables originales

| Columna | Descripcion |
|---|---|
| `Year_Birth` | Anyo de nacimiento del cliente |
| `Education` | Nivel de estudios |
| `Marital_Status` | Estado civil |
| `Income` | Ingresos anuales del hogar |
| `Kidhome` / `Teenhome` | Numero de ninos / adolescentes en el hogar |
| `Dt_Customer` | Anyo de alta en la empresa |
| `MntWines`, `MntMeat`, ... | Gasto en cada categoria de producto (ultimos 2 anyos) |
| `NumWebPurchases`, `NumStorePurchases`, ... | Compras por canal |
| `AcceptedCmp1`–`AcceptedCmp5` | Si se acepto cada campana |
| `Response` | Si se acepto la ultima campana |

---

## Retos y transformaciones de los datos

### 1. Valores nulos y registros invalidos

- 24 filas con `Income` nulo eliminadas
- 3 filas con anyos de nacimiento implausibles eliminadas
- 7 filas con valores de `Marital_Status` no interpretables (`Absurd`, `YOLO`, `Alone`) eliminadas

### 2. Parseo de fechas

`Dt_Customer` estaba almacenado como cadena en formato `%d-%m-%Y`. Parsear sin especificar el formato produjo confusion dia-mes cuando el dia era <= 12, generando 1.311 valores `NaT`. Solucionado pasando `format='%d-%m-%Y'` explicitamente y extrayendo el anyo.

### 3. Deteccion de outliers — Isolation Forest en lugar de IQR

El metodo IQR aplicado secuencialmente por columnas produjo una perdida acumulada de casi el 50% de los datos (2.206 → 827 filas). Isolation Forest evalua los outliers globalmente sobre todas las variables simultaneamente, eliminando solo los registros genuinamente anomalos mientras preserva el tamano del dataset. Con `contamination=0.1`, se eliminaron 221 outliers (~10%), confirmado visualmente mediante la distribucion del anomaly score.

### 4. Feature engineering

Seis nuevas variables derivadas de las existentes:

| Variable | Descripcion |
|---|---|
| `TotalSpend` | Suma del gasto en todas las categorias de producto |
| `TotalPurchases` | Suma de compras en web, catalogo y tienda |
| `TotalChildren` | Suma de `Kidhome` y `Teenhome` |
| `TotalCampaignsAccepted` | Suma de todas las campanas aceptadas |
| `SpendingPerPurchase` | `TotalSpend` / `TotalPurchases` |
| `WebEngagementRate` | `NumWebVisitsMonth` / `NumWebPurchases` |

**Dataset final: 1.985 filas x 32 columnas**

---

## Analisis exploratorio de datos

Hallazgos principales del EDA:

- **Income** esta fuertemente correlacionado con el gasto en todas las categorias de producto (Pearson r > 0.6 en la mayoria)
- **TotalChildren** tiene una fuerte correlacion negativa con ingresos y gasto — las familias con hijos gastan significativamente menos
- **Education** y **Marital_Status** muestran diferencias estadisticamente significativas en ingresos entre grupos (test de Kruskal-Wallis)
- **Las preferencias de canal** varian segun el nivel de ingresos — clientes de ingresos altos usan mas el catalogo; los de ingresos bajos visitan mas la web pero compran menos
- **La respuesta a campanas** se concentra en clientes de ingresos altos; la mayoria de clientes en todos los grupos rechazo todas las campanas

---

## Metodologia

1. **Carga y limpieza de datos** — gestion de nulos, fechas invalidas, categorias ambiguas y outliers
2. **Analisis exploratorio de datos** — distribuciones, correlaciones y tests estadisticos
3. **Feature engineering** — derivacion de variables de comportamiento de alto nivel
4. **Preprocesamiento** — One-Hot Encoding, RobustScaler, PCA
5. **Seleccion de K** — metodo del codo y silueta para K=2 a K=10
6. **Entrenamiento del modelo** — K-Means; GMM y DBSCAN probados y descartados
7. **Evaluacion** — silueta y visualizacion con UMAP
8. **Perfilado de clusters** — `groupby('Cluster').agg()` sobre las variables originales sin escalar

---

## Pipeline de preprocesamiento

| Paso | Transformacion | Motivo |
|---|---|---|
| Codificacion categorica | One-Hot Encoding | K-Means requiere entrada numerica |
| Escalado numerico | `RobustScaler` (mediana + IQR) | Robusto frente a los outliers restantes tras Isolation Forest |
| Reduccion de dimensionalidad | PCA — 15 componentes, 90% de varianza | Aborda la maldicion de la dimensionalidad; mejora la separacion entre clusters |

Todos los pasos ajustados sobre el dataset completo (no supervisado — no requiere division train/test).

---

## Seleccion del modelo

Se evaluaron tres algoritmos de clustering:

| Modelo | Silueta | Notas |
|---|---|---|
| K-Means K=3 (variables originales) | 0.20 | Linea base |
| GMM K=6 (minimo BIC) | 0.04 | Peor que K-Means; descartado |
| DBSCAN (eps=5) | — | Produjo 1 cluster + 3 puntos de ruido; descartado |
| **K-Means K=3 + PCA** | **0.24** | Mejor resultado; modelo final |

**Seleccion de K:** La silueta fue practicamente plana para K=2–10 incluso con PCA. El metodo del codo mostro la inflexion mas pronunciada en K=3. K=2 produjo una silueta media mas alta (0.37) pero el cluster 0 tenia muchos puntos mal asignados (coeficientes negativos), haciendolo poco fiable. K=4 empeoro a 0.21 con mucho solapamiento en UMAP.

---

## Segmentos de clientes

### Cluster 1 — Cliente Premium

**Demografía:** Nacido ~1968 | Casado | Ingresos ~€73k | Casi sin hijos (0.26 media)  
**Gasto:** Gasto total €1.274 media | Gasto por compra €69 | Categorias principales: vinos y carne  
**Canales:** Prefiere el catalogo | Menores compras con descuento (1.42) — compra a precio completo  
**Campanas:** Mas receptivo (0.79 campanas aceptadas media)  
**Recomendacion:** Programa de fidelizacion VIP con recompensas exclusivas y acceso anticipado a nuevos productos. Dirigir campanas personalizadas aqui — este segmento ya responde y tiene presupuesto para aumentar el ticket medio.

---

### Cluster 0 — Cazador de Ofertas

**Demografía:** Nacido ~1966 | Casado | Ingresos ~€57k | ~1 hijo  
**Gasto:** Gasto total €634 media | Gasto por compra €39  
**Canales:** Tienda fisica y web | Mayor compra con descuento (3.44) — las promociones impulsan la conversion  
**Campanas:** Respuesta moderada (0.42 campanas aceptadas media)  
**Recomendacion:** Campanas de descuento segmentadas y promociones de tiempo limitado. Aqui es donde el presupuesto promocional tiene mayor ROI — los descuentos generan compras incrementales que no se producen a precio completo.

---

### Cluster 2 — Escaparatista

**Demografía:** Nacido ~1972 (mas joven) | Casado | Ingresos ~€34k (mas bajos) | Mas hijos (1.25 media)  
**Gasto:** Gasto total €80 media | Gasto por compra €13  
**Canales:** Mayor frecuencia de visitas web (6.43/mes) y tasa de engagement web (3.86) — navega mucho pero rara vez convierte  
**Campanas:** Respuesta muy baja (0.17 campanas aceptadas media)  
**Recomendacion:** Promociones en web: packs familiares, ofertas 2x1, productos de entrada a bajo precio. El alto engagement web indica intencion de compra — la barrera es el presupuesto, no el interes.

---

## Limitaciones del modelo

- **Silueta moderada (0.24):** Los perfiles de cliente en este dataset transicionan de forma gradual — no hay fronteras naturales nitidas entre segmentos. Es una caracteristica del dato, no un fallo del modelo.
- **K-Means asume clusters esfericos:** El algoritmo usa distancia euclidea y clusters de varianza similar, lo que puede no reflejar la geometria real de los datos de clientes.
- **PCA reduce la interpretabilidad:** Las asignaciones de cluster se calculan sobre componentes principales. El perfilado debe hacerse sobre las variables originales a posteriori.
- **Instantanea estatica:** El dataset representa un momento en el tiempo. El comportamiento del cliente evoluciona — los segmentos identificados hoy pueden cambiar.
- **Sin ground truth:** Sin datos etiquetados no hay forma de validar si los clusters corresponden a grupos reales mas alla de la interpretacion de negocio.
- **La contaminacion de Isolation Forest es una suposicion:** `contamination=0.1` fue validado visualmente mediante la distribucion del anomaly score pero no deriva de conocimiento de dominio.

---

## Conclusiones

Este proyecto identifica tres segmentos de clientes con niveles de ingresos, patrones de gasto y respuesta a campanas claramente diferentes. La segmentacion no es perfecta — una silueta de 0.24 refleja la naturaleza gradual de los perfiles en este dataset — pero los tres segmentos son interpretables y accionables.

El hallazgo mas importante no es una metrica sino una conclusion estrategica: **aplicar la misma campana a todos los clientes es ineficiente**. Los clientes Premium compran sin descuentos; ofrecerselos malgasta presupuesto. Los Cazadores de Ofertas necesitan incentivos de precio para convertir; no darselos pierde ingresos. Los Escaparatistas no estan desenganchados — visitan la web con frecuencia — pero necesitan un punto de entrada con menor friccion economica.

Los tres segmentos tienen implicaciones estrategicas concretas. Los **clientes Premium** registran el mayor gasto por compra (€69 media) y la mayor tasa de aceptacion de campanas (0.79 media), pero compran a precio completo de todos modos — dirigirles descuentos no genera ingresos incrementales. Los **Cazadores de Ofertas** muestran el patron contrario: la mayor frecuencia de compra con descuento (3.44 media) confirma que los incentivos de precio son la palanca que impulsa su conversion. Los **Escaparatistas** visitan la web 6.4 veces al mes de media — la cifra mas alta de los tres segmentos — pero gastan solo €80 en total. Esa brecha entre engagement y gasto es la oportunidad de negocio: promociones web dirigidas eliminan la barrera presupuestaria para un segmento que ya muestra intencion de compra.

La implicacion practica es que una campana identica para todos los clientes es la peor asignacion posible del presupuesto de marketing para los tres segmentos simultaneamente. Este analisis proporciona la capa de segmentacion que hace posibles las estrategias diferenciadas.

---

## Posibles mejoras

- **Clustering jerarquico (Agglomerative):** No asume clusters esfericos y produce un dendrograma que permite elegir K de forma mas intuitiva.
- **Seleccion de variables antes de PCA:** Eliminar variables de baja varianza o muy correlacionadas antes de la reduccion de dimensionalidad podria mejorar la compacidad de los clusters.
- **Datos temporales:** Anadir frecuencia de compra en el tiempo o antiguedad del cliente permitiria segmentaciones mas matizadas — por ejemplo, identificar clientes Premium en riesgo de baja.
- **Clustering ensemble:** Combinar multiples ejecuciones con etiquetas de consenso produce segmentos mas estables, reduciendo la sensibilidad a la inicializacion aleatoria.
- **Validacion externa:** Vincular las etiquetas de cluster a resultados de negocio reales (tasa de conversion, ingresos por cliente, tasa de baja) permitiria una evaluacion mas alla de las metricas internas.
- **Mejor ajuste de DBSCAN:** Una busqueda mas exhaustiva sobre eps y min_samples podria dar mejores resultados que el intento inicial con el grafico k-distancia.

---

## Requisitos

```bash
pip install kagglehub pandas matplotlib seaborn scikit-learn numpy umap-learn
```

---

*Fuente de datos: https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis*
