# Pipeline de Predicción de Viabilidad de Startups — Resumen Ejecutivo

**Objetivo:** clasificar `viabilidad ∈ {0,1}` usando variables tabulares y una variante de texto con la columna `descripcion` (en pipelines separados).

**Flujo:** `exploracion.ipynb` → `/data/processed/startups_sintetico_1000_processed.csv` → `modelado.ipynb` (modelo tabular) y `modelo_texto.ipynb` (solo texto).

**Checklist:**
1) Cargar crudo desde `/data/raw/datos_raw_startups_2025.csv` y limpiar (strings, parseos, flags, outliers).  
2) Derivar features (`intensidad_redes`, `log_monto`, `antiguedad`, `antiguedad_bucket`, `sector_x_estado`, ratios) y One-Hot.  
3) Guardar **solo** el procesado en `/data/processed`.  
4) Modelado tabular: baselines (Dummy, LogReg), candidatos (DT, RF, HistGB), tuning (RF/HGB), selección por AUC, threshold sweep y permutation importance.  
5) Modelo de texto: `TF-IDF(≤800 1–2gram) + LogReg`.  
6) Guardar: `models/modelo_tabular.joblib` y `models/modelo_texto.joblib`.

**Tabla (rellenar desde `modelado.ipynb`):**

| Modelo | Hiperparámetros clave | AUC (val) | F1 (val) | Acc (val) | Prec (val) | Rec (val) |
|---|---|---:|---:|---:|---:|---:|
| Dummy | strategy=stratified |  |  |  |  |  |
| LogReg | C, max_iter=1000 |  |  |  |  |  |
| DecisionTree | max_depth, min_samples_split |  |  |  |  |  |
| RandomForest(best) | n_estimators, max_depth |  |  |  |  |  |
| HistGB(best) | max_depth, learning_rate |  |  |  |  |  |

**Insights esperados:** 
- `log_monto`, `tamano_equipo`, `intensidad_redes` suelen dominar la importancia.  
- `antiguedad_bucket` y cruces como `sector_x_estado_*` ayudan a capturar contexto.

**Recomendaciones:** 
- Ajustar umbral por F1 si se prioriza balance precisión/recuperación.  
- Escalar únicamente para modelos lineales (ya contemplado en pipeline).  
- Tratar outliers de `presencia_redes` (cap a 50 para `intensidad_redes`).  
- Vigilar desbalance y no mezclar `descripcion` con tabular.

**Limitaciones y futuro:** dataset sintético (patrones simplificados); ampliar tamaño, features reales (tracción, unit economics), validación temporal y calibración.