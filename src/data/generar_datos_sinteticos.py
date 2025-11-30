# src/data/generar_datos_sinteticos.py
from __future__ import annotations
from pathlib import Path
import argparse

import numpy as np
import pandas as pd

from src.utils.helpers import parse_money_to_float  # reutilizamos tu helper

RANDOM_STATE = 42


def _choice_non_nan(series: pd.Series, rng: np.random.Generator):
    vals = series.dropna().values
    if len(vals) == 0:
        return np.nan
    return rng.choice(vals)


def _sample_row_for_label(
    df_label: pd.DataFrame,
    label: int,
    rng: np.random.Generator,
) -> dict:
    """
    Genera una fila sintética para una clase (viabilidad = 0 o 1),
    muestreando columnas desde el subset correspondiente y aplicando
    ruido suave en las variables numéricas.
    """
    row: dict = {}

    # --- categóricas directas ---
    row["sector"] = _choice_non_nan(df_label["sector"], rng)
    row["ubicacion"] = _choice_non_nan(df_label["ubicacion"], rng)
    row["inversores_destacados"] = _choice_non_nan(df_label["inversores_destacados"], rng)
    row["tiempo_fundacion"] = _choice_non_nan(df_label["tiempo_fundacion"], rng)
    row["estado_operativo"] = _choice_non_nan(df_label["estado_operativo"], rng)

    # --- monto_financiado: parseamos, metemos factor y devolvemos como string ---
    m_str = _choice_non_nan(df_label["monto_financiado"], rng)
    m_val = parse_money_to_float(m_str)
    if np.isnan(m_val):
        row["monto_financiado"] = m_str
    else:
        # ruido moderado, acotado
        factor = float(np.clip(rng.normal(1.0, 0.35), 0.4, 1.8))
        new_m = max(50.0, m_val * factor)
        row["monto_financiado"] = f"{round(new_m, 1)}"

    # --- num_rondas ---
    nr = _choice_non_nan(df_label["num_rondas"], rng)
    if pd.isna(nr):
        row["num_rondas"] = np.nan
    else:
        nr = int(np.clip(round(float(nr) + rng.integers(-1, 2)), 0, 5))
        row["num_rondas"] = float(nr)

    # --- tamano_equipo ---
    te = _choice_non_nan(df_label["tamano_equipo"], rng)
    if pd.isna(te):
        row["tamano_equipo"] = np.nan
    else:
        te = int(np.clip(round(float(te) + rng.normal(0, 3)), 1, 50))
        row["tamano_equipo"] = float(te)

    # --- exp_fundadores ---
    ef = _choice_non_nan(df_label["exp_fundadores"], rng)
    if pd.isna(ef):
        row["exp_fundadores"] = np.nan
    else:
        ef = float(np.clip(float(ef) + rng.normal(0, 0.8), 0.0, 12.0))
        row["exp_fundadores"] = round(ef, 1)

    # --- presencia_redes ---
    pr = _choice_non_nan(df_label["presencia_redes"], rng)
    if pd.isna(pr):
        row["presencia_redes"] = np.nan
    else:
        pr = float(np.clip(float(pr) + rng.normal(0, 8), -10.0, 160.0))
        row["presencia_redes"] = round(pr, 1)

    # --- descripcion: muestreamos de textos REALES de esa clase ---
    row["descripcion"] = _choice_non_nan(df_label["descripcion"], rng)

    # --- etiqueta ---
    row["viabilidad"] = int(label)

    return row


def generar_dataset_sintetico(
    df_base: pd.DataFrame,
    n_target: int = 15000,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Genera un nuevo dataset de tamaño n_target a partir de df_base (1000 filas),
    manteniendo coherencia por clase y agregando ruido razonable.
    Incluye SIEMPRE las filas originales.
    """
    rng = np.random.default_rng(random_state)

    # Aseguramos columnas esperadas
    expected_cols = [
        "sector",
        "monto_financiado",
        "num_rondas",
        "tamano_equipo",
        "exp_fundadores",
        "presencia_redes",
        "ubicacion",
        "inversores_destacados",
        "descripcion",
        "tiempo_fundacion",
        "estado_operativo",
        "viabilidad",
    ]
    df = df_base[expected_cols].copy()

    df_pos = df[df["viabilidad"] == 1].reset_index(drop=True)
    df_neg = df[df["viabilidad"] == 0].reset_index(drop=True)

    if len(df_pos) == 0 or len(df_neg) == 0:
        raise ValueError("Se necesitan ejemplos de ambas clases (viabilidad=0 y viabilidad=1).")

    p_pos = len(df_pos) / len(df)
    print(f"Distribución original de clases: viables={p_pos:.3f}, no_viables={1-p_pos:.3f}")

    # Empezamos con las filas originales
    rows = df.to_dict(orient="records")

    # Vamos generando hasta llegar al tamaño objetivo
    while len(rows) < n_target:
        label = 1 if rng.random() < p_pos else 0
        if label == 1:
            new_row = _sample_row_for_label(df_pos, 1, rng)
        else:
            new_row = _sample_row_for_label(df_neg, 0, rng)
        rows.append(new_row)

    df_new = pd.DataFrame(rows)[expected_cols]
    return df_new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--in",
        dest="in_path",
        default="data/raw/datos_raw_startups_2025.csv",
        help="CSV base (1000 filas).",
    )
    ap.add_argument(
        "--out",
        dest="out_path",
        default="data/raw/datos_raw_startups_2025_15000.csv",
        help="Ruta de salida para el CSV sintético.",
    )
    ap.add_argument(
        "--n",
        dest="n_rows",
        type=int,
        default=15000,
        help="Cantidad total de filas deseadas (incluye las originales).",
    )
    ap.add_argument(
        "--seed",
        dest="seed",
        type=int,
        default=RANDOM_STATE,
        help="Semilla para reproducibilidad.",
    )
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)

    df_base = pd.read_csv(in_path, encoding="utf-8")
    print(f"📥 Dataset base: {in_path} | shape={df_base.shape}")

    df_new = generar_dataset_sintetico(df_base, n_target=args.n_rows, random_state=args.seed)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_new.to_csv(out_path, index=False, encoding="utf-8")
    print(f"✅ Dataset sintético guardado en: {out_path} | shape={df_new.shape}")


if __name__ == "__main__":
    main()
