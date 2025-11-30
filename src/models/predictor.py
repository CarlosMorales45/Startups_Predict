from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import joblib

from src.data.preparar_datos import prepare_dataframe
from src.features.crear_features import build_features
from src.explainability.explicacion import explicar_texto

RANDOM_STATE = 42

# Campos que alimentan al modelo TABULAR (opcionales para el usuario final)
TABULAR_KEYS = [
    "monto_financiado",
    "num_rondas",
    "tamano_equipo",
    "exp_fundadores",
    "presencia_redes",
    "inversores_destacados",
    "tiempo_fundacion",
    "sector",
    "ubicacion",
    "estado_operativo",
]


def _find_project_root(start: Path | None = None) -> Path:
    p = start or Path.cwd()
    for cand in [p, *p.parents]:
        if (cand / "data" / "processed").exists():
            return cand
    raise FileNotFoundError("No se encontró la carpeta 'data/processed'.")


def cargar_modelos(root: Optional[Path] = None):
    """
    Carga el modelo tabular y el modelo de texto desde /models
    y devuelve también la lista de columnas de entrenamiento del modelo tabular.
    """
    root = root or _find_project_root()
    model_tab_path = root / "models" / "modelo_tabular.joblib"
    model_text_path = root / "models" / "modelo_texto.joblib"

    model_tab = joblib.load(model_tab_path)
    model_text = joblib.load(model_text_path)

    # Debe haberse guardado en el entrenamiento tabular
    feature_names_tab = getattr(model_tab, "_feature_names", None)

    return model_tab, model_text, feature_names_tab


def _tiene_info_tabular(raw_inputs: Dict[str, Any]) -> bool:
    """
    Devuelve True si el usuario llenó al menos un campo tabular
    (no None, no cadena vacía).
    """
    for k in TABULAR_KEYS:
        if k in raw_inputs:
            v = raw_inputs[k]
            if v is None:
                continue
            if isinstance(v, str):
                if v.strip() != "":
                    return True
            else:
                # números, bool, etc.
                return True
    return False


def preparar_fila_tabular(
    raw_inputs: Dict[str, Any],
    feature_names: list[str],
) -> pd.DataFrame:
    """
    Toma un diccionario con los mismos campos que el dataset crudo
    y lo transforma a X_tabular compatible con el modelo entrenado.
    """
    # 1. DataFrame crudo con 1 fila
    df_raw = pd.DataFrame([raw_inputs])

    # 2. Limpieza igual que en entrenamiento
    df_clean = prepare_dataframe(df_raw, do_impute=False)

    # 3. Features igual que en entrenamiento
    df_feat = build_features(df_clean)

    # 4. Eliminamos columnas que no son features del modelo tabular
    for col_drop in ["descripcion", "viabilidad"]:
        if col_drop in df_feat.columns:
            df_feat = df_feat.drop(columns=[col_drop])

    # 5. Reindexar a las columnas que el modelo espera
    X = df_feat.reindex(columns=feature_names, fill_value=0.0).astype(float)

    return X


def predecir_viabilidad(
    raw_inputs: Dict[str, Any],
    alpha: float = 0.5,
) -> Dict[str, Any]:
    """
    Usa ambos modelos para predecir la viabilidad de una startup.

    - raw_inputs: diccionario con llaves tipo:
        "descripcion", "monto_financiado", "num_rondas",
        "tamano_equipo", "exp_fundadores", "presencia_redes",
        "inversores_destacados", "tiempo_fundacion",
        "sector", "ubicacion", "estado_operativo"
      (no todas son obligatorias; si no se llenan, el modelo tabular se omite).

    - alpha: peso del modelo tabular en la combinación:
        p_final = alpha * p_tab + (1 - alpha) * p_text
    """
    root = _find_project_root()
    model_tab, model_text, feature_names = cargar_modelos(root)

    descripcion = str(raw_inputs.get("descripcion", "") or "").strip()

    # --- Modelo de TEXTO ---
    proba_text = float(model_text.predict_proba([descripcion])[0, 1])
    thr_text = float(getattr(model_text, "_thr_demo", 0.5))
    pred_text = proba_text >= thr_text

    # --- Modelo TABULAR (solo si hay info tabular suficiente) ---
    proba_tab: Optional[float] = None
    pred_tab: Optional[bool] = None
    usa_tabular = feature_names is not None and _tiene_info_tabular(raw_inputs)

    if usa_tabular:
        try:
            X_tab = preparar_fila_tabular(raw_inputs, feature_names)
            if hasattr(model_tab, "predict_proba"):
                proba_tab = float(model_tab.predict_proba(X_tab)[0, 1])
            else:
                proba_tab = float(model_tab.decision_function(X_tab)[0])
            pred_tab = proba_tab >= 0.5
        except Exception:
            proba_tab = None
            pred_tab = None
            usa_tabular = False

    # --- Probabilidad combinada (promedio ponderado) ---
    if proba_tab is not None:
        proba_comb = float(alpha * proba_tab + (1.0 - alpha) * proba_text)
    else:
        proba_comb = proba_text

    # --- Decisión final: OR entre modelos ---
    pred_comb = bool(
        pred_text or (proba_tab is not None and pred_tab)
    )

    # --- Explicación basada en texto ---
    explic = explicar_texto(model_text, descripcion, thr=thr_text)

    # Ajustar la interpretación para que coincida con la decisión combinada
    if pred_comb != explic.get("viable", False):
        if pred_comb:
            explic["interpretacion"] = (
                f"Aunque el texto por sí solo no muestra todavía mucha tracción "
                f"(probabilidad texto {proba_text:.2f}), los datos numéricos del "
                f"proyecto son suficientemente fuertes, por lo que la predicción "
                f"combinada considera que la startup **sí parece viable**."
            )
        else:
            explic["interpretacion"] = (
                f"Aunque el texto suena prometedor, los datos numéricos aún no "
                f"respaldan la tracción o madurez del proyecto; por eso la "
                f"predicción combinada considera que la startup **no parece viable**."
            )
        explic["viable"] = bool(pred_comb)

    return {
        "proba_texto": proba_text,
        "thr_texto": thr_text,
        "pred_texto": bool(pred_text),
        "proba_tabular": proba_tab,
        "pred_tabular": None if pred_tab is None else bool(pred_tab),
        "proba_combinada": proba_comb,
        "pred_combinada": bool(pred_comb),
        "alpha": float(alpha),
        "usa_tabular": bool(usa_tabular),
        "explicacion_texto": explic,
    }
