# Asistente de Predicción de Viabilidad de Startups

Este repositorio contiene un pipeline completo para entrenar modelos de **viabilidad de startups** (tabular y texto) y una **app web en Streamlit** para hacer predicciones combinando ambos.

---

## 1. Flujo general del proyecto

1. **Exploración y limpieza**
   - Notebook: `exploracion.ipynb`
   - Entrada cruda: `data/raw/datos_raw_startups_2025.csv`
   - Salida limpia: `data/processed/clean_base.parquet`
   - Módulo principal: `src/data/preparar_datos.py`  
     - Normaliza strings (`sector`, `ubicacion`, `estado_operativo`).
     - Parsea números y fechas (`monto_financiado`, `num_rondas`, `tiempo_fundacion`, etc.).
     - Crea *flags* de ausencia `flag_na_*` para valores faltantes.

2. **Ingeniería de features tabulares**
   - Notebook: `modelado.ipynb`
   - Entrada: `data/processed/clean_base.parquet`
   - Salida procesada: `data/processed/startups_sintetico_1000_processed.csv`
   - Módulo: `src/features/crear_features.py`  
     - `intensidad_redes` (normalización de 0–100).
     - `log_monto` (transformación log1p del monto).
     - `antiguedad` y `antiguedad_bucket` (0–1, 2–3, 4–7, 8+ años).
     - Ratios como `ratio_equipo_inversion` y `exp_media_fundadores`.
     - Cruces como `sector_x_estado`.
     - One-hot encoding de `sector`, `ubicacion`, `estado_operativo`, `antiguedad_bucket`, `sector_x_estado`.

3. **Modelado**
   - Script de entrenamiento: `src/models/entrenar_modelo.py`
   - Dataset procesado: `data/processed/startups_sintetico_1000_processed.csv`
   - Salidas:
     - Modelo tabular: `models/modelo_tabular.joblib`
     - Modelo de texto: `models/modelo_texto.joblib`
     - Reporte tabular (opcional): `models/report_tabular.json`

   ### 3.1 Modelo tabular (`--mode tabular`)

   - Baselines: `DummyClassifier`, `LogisticRegression` (con `StandardScaler`).
   - Modelos candidatos: `DecisionTreeClassifier`, `RandomForestClassifier`, `HistGradientBoostingClassifier`.
   - `RandomForest` con `GridSearchCV` sobre:
     - `n_estimators`, `max_depth`, `min_samples_split`.
   - `HistGradientBoosting` con `RandomizedSearchCV` sobre:
     - `max_depth`, `learning_rate`, `max_bins`.
   - Selección por **AUC de validación**.
   - Reentrenamiento en `train+valid` y evaluación en `test` (70/15/15).
   - Se guarda `final._feature_names` para reproducir el preprocesado en predicción.

   **Tabla (rellenar desde las salidas de entrenamiento):**

   | Modelo             | Hiperparámetros clave              | AUC (val) | F1 (val) | Acc (val) | Prec (val) | Rec (val) |
   |--------------------|-------------------------------------|----------:|---------:|----------:|-----------:|----------:|
   | Dummy              | strategy=stratified                |          |         |          |           |          |
   | LogReg             | C, max_iter=1000                   |          |         |          |           |          |
   | DecisionTree       | max_depth, min_samples_split       |          |         |          |           |          |
   | RandomForest(best) | n_estimators, max_depth            |          |         |          |           |          |
   | HistGB(best)       | max_depth, learning_rate, max_bins |          |         |          |           |          |

   ### 3.2 Modelo de texto (`--mode texto`)

   - Pipeline: `TfidfVectorizer + LogisticRegression (class_weight='balanced')`.
   - `TfidfVectorizer` configurado con:
     - `max_features=1500`, `ngram_range=(1, 2)`.
     - `token_pattern` que acepta números/unidades (`25k`, `12%`, `usd`, etc.).
     - `strip_accents="unicode"`.
     - `stop_words` en español.
     - `preprocessor` = `_augment_text_with_numeric_signals`  
       (añade tags como `__monto_alto__`, `__clientes_altos__`, `__crecimiento_medio__`, etc.).
   - Split 70/15/15 y **búsqueda de umbral** en validación:
     - Se recorre `thr ∈ [0.1, 0.9]` y se calcula F1, precisión y recall.
     - Se obtiene:
       - `thr_f1`: umbral con mejor F1.
       - `thr_demo`: umbral más estricto con precisión ≥ 0.8 (si existe); si no, se usa `thr_f1`.
   - Se reentrena en `train+valid` y se evalúa en `test`.
   - Se guardan en el pipeline:
     - `pipe._thr_demo` (umbral recomendado).
     - `pipe._thr_f1` y una grilla de métricas por umbral (`pipe._thr_grid`).

---

## 2. Explicabilidad

Módulo: `src/explainability/explicacion.py`

- Para modelo tabular: función `permutation_importance_top` para obtener las features más relevantes por AUC (permute & drop score).
- Para modelo de texto:
  - Se localiza `TfidfVectorizer` y el clasificador dentro del Pipeline.
  - Se calculan contribuciones término a término (`TF-IDF × coef_`).
  - Se muestran solo términos con contribución positiva y se traducen a mensajes legibles (`mrr` → “ingresos recurrentes (MRR)”, etc.).
  - `explicar_texto` devuelve:
    - `probabilidad`, `umbral`, `viable` (True/False).
    - `justificacion` (tokens con peso).
    - `interpretacion` en lenguaje natural para exposición.

---

## 3. Predicción combinada (texto + tabular)

Módulo: `src/models/predictor.py`

- Carga los modelos:
  - `models/modelo_tabular.joblib`
  - `models/modelo_texto.joblib`
- Reconstruye el pipeline tabular a partir de los mismos pasos que en entrenamiento:
  1. `prepare_dataframe` (limpieza y flags `flag_na_*`).
  2. `build_features` (features numéricas, cruces, one-hot).
  3. Reindexado a `model_tab._feature_names` con `fill_value=0.0`.

- Lógica:
  - Siempre calcula `proba_texto` y aplica el umbral `thr_texto` (desde `_thr_demo` o 0.5).
  - Sólo usa el modelo tabular si hay al menos 1 campo tabular rellenado.
  - Si hay probabilidad tabular (`proba_tabular`), combina así:

    p_final = alpha * p_tabular + (1 - alpha) * p_texto

  - Devuelve un diccionario con:
    - `proba_texto`, `thr_texto`, `pred_texto`
    - `proba_tabular`, `pred_tabular`
    - `proba_combinada`, `pred_combinada`
    - `alpha`, `usa_tabular`
    - `explicacion_texto` (interpretación y tokens relevantes).

---

## 4. App web con Streamlit

Archivo principal: `app_streamlit.py` (nombre de ejemplo).

- Formulario con tres bloques:
  1. **Descripción** (obligatoria).
  2. **Datos tabulares** (opcionales: monto, rondas, tamaño del equipo, etc.).
  3. **Peso del modelo tabular (α)** con un slider entre 0 y 1.
- Llama a `predecir_viabilidad(raw_inputs, alpha)` y muestra:
  - Métricas:
    - Probabilidad (texto) + etiqueta “Viable / No viable”.
    - Probabilidad (tabular) si se usó.
    - Probabilidad combinada y predicción final.
  - Explicación basada en el texto y *tokens* que más influyeron.
  - Entrada enviada al modelo, dentro de un `st.expander`.

### 4.1 Ejecución local

```bash
# Crear y activar entorno virtual (opcional)
python -m venv .venv
source .venv/bin/activate  # En Windows: .venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Lanzar la app
streamlit run app_streamlit.py
```

---

## 5. Evaluación de modelos

Script: `src/models/evaluar_modelo.py`

- Modo tabular:

```bash
python -m src.models.evaluar_modelo   --mode tabular   --model models/modelo_tabular.joblib
```

- Modo texto:

```bash
python -m src.models.evaluar_modelo   --mode texto   --model models/modelo_texto.joblib
```

Genera métricas (AUC, F1, precisión, recall, matrices de confusión) y gráficos ROC/PR en `models/eval/`.

---

## 6. Limitaciones y trabajo futuro

- Dataset **sintético** (patrones simplificados y poco ruido real).
- Faltan variables reales de negocio (MRR real, churn, unit economics, cohortes).
- No hay validación temporal ni calibración probabilística explícita.
- Futuras mejoras posibles:
  - Más datos reales y balance entre clases.
  - Features adicionales de tracción y producto.
  - Ajustes finos del umbral según criterio del negocio (por ejemplo, priorizar falsos negativos vs falsos positivos).
  - Explicabilidad tabular más avanzada (SHAP, LIME) para producción.
