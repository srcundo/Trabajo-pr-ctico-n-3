# Trabajo Práctico N°3 — Análisis de sentimiento en tweets (NLP)

Proyecto de análisis de sentimiento sobre el dataset Sentiment140 (1.6 millones de
tweets). Se entrenaron y evaluaron dos modelos de clasificación, se compararon
contra un modelo pre-entrenado (TextBlob) y se analizaron los errores usando
similitud coseno. Como extra, se armó un minijuego interactivo que usa el modelo
entrenado para clasificar respuestas del usuario en tiempo real.

---

## 1. Datos

Se usaron los dos archivos provistos en la consigna:

| Archivo | Filas | Descripción |
|---|---|---|
| `training.1600000.processed.noemoticon.csv` | 1.600.000 | Tweets etiquetados automáticamente según el emoticono que tenían (después se les borró el emoticono). Solo clases 0 (negativo) y 4 (positivo), perfectamente balanceado. |
| `testdata.manual.2009.06.14.csv` | 498 | Tweets etiquetados a mano. Tiene las 3 clases: 0 (negativo), 2 (neutral) y 4 (positivo). |

Columnas de ambos archivos:

| Columna | Descripción |
|---|---|
| `polarity` | Sentimiento del tweet (0 = negativo, 2 = neutral, 4 = positivo) |
| `id` | ID del tweet |
| `date` | Fecha de publicación |
| `query` | Término de búsqueda con el que se obtuvo el tweet (NO_QUERY si no hubo) |
| `user` | Usuario que lo publicó |
| `text` | Texto del tweet |

**Nota importante**: el CSV de entrenamiento pesa 228 MB y supera el límite de
GitHub, por eso no está en el repositorio. Se puede descargar desde el link de la
consigna.

---

## 2. Estructura del proyecto

```
Trabajo práctico 3/
├── data/                  (los dos CSV — el de training no se sube a GitHub)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocesamiento.ipynb
│   ├── 03_modelado_baseline.ipynb
│   ├── 04_optimizacion.ipynb
│   ├── 05_comparacion_textblob.ipynb
│   ├── 06_evaluacion_final.ipynb
│   └── 07_juego_cita.ipynb
├── modelos/               (vectorizadores, matrices y modelos entrenados — se generan al correr las notebooks)
├── .gitignore
└── README.md
```

Cada notebook carga lo que generó la anterior desde la carpeta `modelos/`, así no
se reprocesa nada y todas trabajan con la misma versión de los datos.

---

## 3. Metodología

**01 — EDA**: análisis exploratorio. Nulos, duplicados, balance de clases,
longitud de tweets, palabras más frecuentes por clase y tópicos del test. Acá
apareció el hallazgo central: el training es binario pero el test tiene 3 clases.

**02 — Preprocesamiento**: limpieza de texto con regex (URLs, menciones, caracteres
especiales) + stopwords de nltk. Vectorización con Bag of Words y TF-IDF, limitando
el vocabulario a las 20.000 palabras más frecuentes. Los vectorizadores se ajustan
solo con train para no filtrar información del test.

**03 — Modelado baseline**: los dos modelos pedidos por la consigna, con
hiperparámetros por defecto. MultinomialNB con Bag of Words y LogisticRegression
con TF-IDF. Evaluados sobre train y test.

**04 — Optimización**: búsqueda de hiperparámetros con RandomizedSearchCV sobre una
muestra de 200 mil tweets, y reentrenamiento con el training completo usando los
mejores valores encontrados. Incluye chequeo de overfitting.

**05 — Comparación con TextBlob**: el modelo pre-entrenado que pide la consigna.
Se evaluó dos veces: sobre el test completo (3 clases) y sobre el subset binario
(mismos 359 tweets que los otros modelos, para que la comparación sea justa).

**06 — Evaluación final**: tabla comparativa de los 3 métodos, evaluación completa
train vs test con matrices de confusión, ROC-AUC, similitud coseno sobre los
errores del modelo (la métrica obligatoria de la consigna) y conclusiones.

**07 — Juego**: simulador de cita romántica. Hace preguntas incómodas, el usuario
responde en español, la respuesta se traduce al inglés y el modelo entrenado
predice si suena positiva, negativa o dudosa según la probabilidad.

---

## 4. Decisiones tomadas durante el desarrollo

- **Evaluación sobre el subset binario del test**: el modelo nunca vio la clase
  neutral en el entrenamiento, así que no puede predecirla. Se filtraron los 139
  neutrales del test y se evaluó sobre los 359 restantes.
- **Regex + nltk en vez de spacy para limpiar**: con 1.6M de tweets, spacy fila
  por fila tardaba demasiado. En textos tan cortos la lematización aporta poco.
- **max_features=20000 en los vectorizadores**: sin ese límite el vocabulario
  explota a cientos de miles de términos, la mayoría typos o palabras que aparecen
  una sola vez.
- **Búsqueda de hiperparámetros sobre una muestra**: RandomizedSearchCV con
  cross-validation sobre 1.6M filas era demasiado costoso. Se buscó sobre 200 mil
  (estratificado) y se reentrenó con todo.
- **Limpieza distinta para TextBlob**: la limpieza de los modelos saca stopwords,
  y "not"/"no" están en esa lista — eso rompe negaciones como "not good", que
  TextBlob sí sabe interpretar. Para TextBlob solo se sacaron URLs y menciones.
- **Traducción en el juego**: el modelo se entrenó en inglés, así que las
  respuestas en español se traducen con deep-translator antes de clasificar.

---

## 5. Resultados

| Modelo | Accuracy (test) | AUC |
|---|---|---|
| Naive Bayes (BoW) | 0.83 | 0.90 |
| Logistic Regression (TF-IDF) | 0.82 | 0.90 |
| TextBlob (pre-entrenado) | 0.64 | — |

---

## 6. Conclusiones principales

1. **Naive Bayes le ganó a Logistic Regression** (0.83 vs 0.82). Con tweets tan
   cortos (mediana de 69 caracteres), TF-IDF no tiene margen para diferenciarse
   de un conteo simple de palabras.
2. **La optimización de hiperparámetros casi no cambió nada**: con 1.6M de
   ejemplos balanceados, los modelos ya generalizaban bien con los valores por
   defecto.
3. **No hay overfitting — el test dio mejor que el train**. La explicación está
   en las etiquetas: las del training se pusieron automáticamente por emoticono
   (método ruidoso), las del test las puso una persona a mano.
4. **TextBlob rindió peor (0.64) pero no por estar mal calibrado**: predijo
   "neutral" en ~17% de tweets que eran claramente positivos o negativos. Es más
   conservador — puede dudar, cosa que nuestros modelos no pueden hacer.
5. **El análisis con similitud coseno mostró un límite del preprocesamiento**: los
   tweets con negaciones ("No need for remorse!") pierden el "not"/"no" al sacar
   stopwords y el modelo termina prediciendo el sentimiento contrario. La
   optimización no arregla esto: el problema pasa antes de que el modelo vea el
   texto.

---

## 7. Cómo reproducir

1. Descargar los dos CSV y subirlos a la carpeta `data/` en Google Drive
   (estructura de carpetas en la sección 2).
2. Abrir las notebooks en Google Colab en orden (01 → 07). Cada una monta Drive
   al principio y pide autorización.
3. Correr cada notebook completa con "Entorno de ejecución → Ejecutar todo".
   Las notebooks van guardando lo procesado en `modelos/`, así que hay que
   respetar el orden.

Librerías: todas vienen preinstaladas en Colab salvo `deep-translator` (solo para
la notebook 07, se instala con una celda de pip incluida ahí).

---

## Tecnologías utilizadas

- Python 3 (Google Colab)
- pandas / numpy
- matplotlib / seaborn
- scikit-learn
- nltk
- TextBlob
- deep-translator (para el juego)
