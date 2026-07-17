# Trabajo Práctico N°3 — Análisis de sentimiento en tweets (NLP)

Análisis de sentimiento sobre el dataset Sentiment140 (1.6 millones de tweets).
Se entrenaron dos modelos de clasificación, se compararon contra un modelo
pre-entrenado (TextBlob) y se analizaron los errores con similitud coseno. Como
extra, hay un minijuego interactivo que usa el modelo entrenado en tiempo real.

Todos los archivos: https://drive.google.com/drive/folders/1vEbizVh6DDI1zpShmBDvno-dIn-IfmZd?usp=sharing

---

## 1. Datos

| Archivo | Filas | Descripción |
|---|---|---|
| `training.1600000.processed.noemoticon.csv` | 1.600.000 | Tweets etiquetados automáticamente según el emoticono que tenían (después se les borró). Solo clases 0 (negativo) y 4 (positivo). |
| `testdata.manual.2009.06.14.csv` | 498 | Tweets etiquetados a mano. Tiene las 3 clases: 0, 2 (neutral) y 4. |

De ese 1.6M, se separó un **15% como validación** (240k tweets), quedando 1.36M
para entrenar. La validación nunca se usa para entrenar — solo para medir.

Columnas: `polarity` (target), `id`, `date`, `query`, `user`, `text`.

**Nota**: el CSV de training pesa 228 MB y supera el límite de GitHub, así que no
está en el repositorio. Se descarga del link de la consigna.

---

## 2. Estructura del proyecto

```
Trabajo práctico 3/
├── data/
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocesamiento.ipynb
│   ├── 03_modelado_baseline.ipynb
│   ├── 04_optimizacion.ipynb
│   ├── 05_comparacion_textblob.ipynb
│   ├── 06_evaluacion_final.ipynb
│   └── 07_juego_cita.ipynb
├── modelos/
├── .gitignore
└── README.md
```
---

## 3. Por qué se usan tres conjuntos (train / validación / test)

Al principio solo estaba el training (1.6M) y el test manual (359 tweets
binarios). El problema: 359 es poco para confiar en el número. Por eso se separó
una validación grande (240k) del training, sin tocar el test manual.

- **Train (1.36M)**: entrena los modelos.
- **Validación (240k)**: mide qué tan bien generalizan, con volumen grande. Es la
  métrica principal del proyecto.
- **Test manual (359 binarios)**: se usa para tres cosas puntuales que la
  validación no puede: compararse con TextBlob (pide etiquetas reales), como
  segunda validación con etiquetas de otro origen (puestas a mano), y para el
  análisis de similitud coseno (necesita confiar en que la etiqueta real es
  correcta).

---

## 4. Metodología por notebook

**01 — EDA**: nulos, duplicados, balance de clases, longitud de tweets, palabras
frecuentes. Hallazgo clave: el training no tiene tweets neutrales, pero el test sí.

**02 — Preprocesamiento**: limpieza con regex + stopwords de nltk. Se separa la
validación (85/15, estratificada). Vectorización con Bag of Words y TF-IDF
(vocabulario de 20.000 palabras), ajustada solo con train.

**03 — Modelado baseline**: Naive Bayes (BoW) y Logistic Regression (TF-IDF), sin
ajustar nada. Se evalúan sobre train, validación y test manual, para tener un
punto de partida y chequear overfitting desde el principio.

**04 — Optimización**: búsqueda de hiperparámetros con RandomizedSearchCV, usando
`PredefinedSplit` para entrenar siempre con train y validar siempre contra la
validación real (sin muestrear ni repetir folds). El modelo final se entrena solo
con train, dejando la validación limpia para medir.

**05 — Comparación con TextBlob**: TextBlob evaluado sobre el test manual (el
único con etiquetas reales para comparar). Se mide dos veces: con las 3 clases
completas, y sobre el mismo subset binario que NB y LR.

**06 — Evaluación final**: acá se junta todo. La comparación NB vs LR sobre
validación (la métrica que importa), matrices de confusión de las tres
particiones, ROC-AUC sobre validación, similitud coseno sobre errores del test
manual (métrica obligatoria de la consigna), y conclusiones.

**07 — Juego**: simulador de cita en español. Traduce la respuesta al inglés,
usa el modelo entrenado para clasificarla como positiva, negativa o dudosa.

---

## 5. Decisiones clave

- **Se agregó una validación separada** porque 359 tweets de test daban números
  poco confiables — con 240k, el resultado no depende de qué tweets puntuales
  tocaron en la muestra.
- **PredefinedSplit en vez de cross-validation**: al tener una validación real,
  no hace falta repetir folds ni muestrear — se entrena una vez contra train,
  se mide una vez contra validación.
- **El modelo final se entrena solo con train**, no con train+validación
  combinados, para que la validación quede siempre disponible como medida limpia.
- **Limpieza distinta para TextBlob**: la limpieza normal saca "not"/"no" como
  stopwords, lo cual rompe negaciones. Para TextBlob se usa una limpieza más
  liviana que las conserva.
- **max_features=20000** en los vectorizadores, para no explotar el vocabulario
  con typos y palabras que aparecen una sola vez.

---

## 6. Resultados

| Modelo | Accuracy (validación, 240k) | AUC (validación) |
|---|---|---|
| Naive Bayes (BoW) | 0.77 | 0.84 |
| Logistic Regression (TF-IDF) | 0.78 | 0.86 |

Sobre el test manual (359, secundario): NB 0.82, LR 0.81, TextBlob 0.64.

---

## 7. Conclusiones

- Sobre validación, LR le gana a NB en accuracy y en AUC. Es un resultado
  distinto al que daba el test chico (donde NB ganaba) — con más datos, el
  resultado es más confiable.
- Optimizar hiperparámetros no mejoró nada: mismo accuracy que el baseline.
- No hay overfitting: train y validación dan resultados muy parecidos.
- NB comete más falsos negativos (cauteloso), LR más falsos positivos
  (generoso) — patrón que se repite en train y validación.
- El test manual da mejor accuracy que train/validación en los dos modelos,
  probablemente porque sus etiquetas las puso una persona, mientras que las de
  train/validación se pusieron automático por emoticono (más ruidosas).
- TextBlob rinde peor (0.64) pero no por estar mal calibrado — predijo
  "neutral" en ~17% de tweets binarios, algo que NB y LR no pueden hacer.
- Similitud coseno: en 2 de 3 errores de LR analizados, tweets con negación
  pierden el "not"/"no" en la limpieza, y el modelo termina prediciendo el
  sentimiento contrario. El problema es del preprocesamiento, no del modelo —
  se sostuvo igual con una configuración de entrenamiento distinta.

---

## 8. Cómo reproducir

1. Descargar los CSV y subirlos a `data/` en Google Drive.
2. Abrir las notebooks en Colab, en orden (01 a 07).
3. Correr cada una completa con "Entorno de ejecución → Ejecutar todo".

Librerías: todas vienen en Colab salvo `deep-translator` (solo para la notebook
07, se instala con una celda de pip incluida ahí).

---

## Tecnologías

Python 3 (Colab), pandas, numpy, matplotlib, seaborn, scikit-learn, nltk,
TextBlob, deep-translator.
