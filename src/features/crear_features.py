# src/features/crear_features.py
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from datetime import datetime

NOW_YEAR = datetime.now().year

NUM_IMPUTE_COLS = [
    "monto_financiado","num_rondas","tamano_equipo","exp_fundadores",
    "presencia_redes","inversores_destacados","tiempo_fundacion"
]

CAT_ENCODE_COLS = ["sector","ubicacion","estado_operativo","antiguedad_bucket"]

def _impute_numeric(df):
    for c in NUM_IMPUTE_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())
    return df

def _make_intensidad_redes(df):
    # cap a 50 y reescala 0-1
    if "presencia_redes" in df.columns:
        capped = df["presencia_redes"].clip(lower=0, upper=100)
        df["intensidad_redes"] = (np.minimum(capped, 50) / 50.0).round(4)
    return df

def _make_log_monto(df):
    if "monto_financiado" in df.columns:
        df["log_monto"] = np.log1p(df["monto_financiado"]).round(6)
    return df

def _make_antiguedad(df):
    if "tiempo_fundacion" in df.columns:
        df["antiguedad"] = (NOW_YEAR - df["tiempo_fundacion"]).clip(lower=0)
        bins = [-1,1,3,7,100]  # 0-1; 2-3; 4-7; 8+ años
        labels = ["0-1","2-3","4-7","8+"]
        df["antiguedad_bucket"] = pd.cut(df["antiguedad"], bins=bins, labels=labels)
    return df

def _make_ratios(df):
    if {"tamano_equipo","monto_financiado"}.issubset(df.columns):
        df["ratio_equipo_inversion"] = (
            df["tamano_equipo"] / (1.0 + (df["monto_financiado"] / 1e5))
        ).round(6)
    if "exp_fundadores" in df.columns:
        # si no hay número de fundadores, asumimos que es media ya
        df["exp_media_fundadores"] = df["exp_fundadores"].astype(float).round(2)
    return df

def _make_cross(df):
    if {"sector","estado_operativo"}.issubset(df.columns):
        df["sector_x_estado"] = (df["sector"].astype(str) + "_" + df["estado_operativo"].astype(str))
    return df

def _encode_categoricals(df):
    for c in CAT_ENCODE_COLS:
        if c in df.columns:
            df[c] = df[c].astype("category")
    df_encoded = pd.get_dummies(df, columns=[c for c in CAT_ENCODE_COLS if c in df.columns], drop_first=False)
    return df_encoded

def build_features(df_clean):
    df = df_clean.copy()
    df = _impute_numeric(df)
    df = _make_intensidad_redes(df)
    df = _make_log_monto(df)
    df = _make_antiguedad(df)
    df = _make_ratios(df)
    df = _make_cross(df)
    # No codificamos 'descripcion' aquí (quedará como columna cruda para la variante de texto)
    df_final = _encode_categoricals(df)
    return df_final

def run(in_path, out_path):
    df = pd.read_parquet(in_path) if in_path.endswith(".parquet") else pd.read_csv(in_path)
    df_proc = build_features(df)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df_proc.to_csv(out_path, index=False, encoding="utf-8")
    return df_proc

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/processed/clean_base.parquet")
    ap.add_argument("--out", dest="out_path", default="data/processed/startups_sintetico_1000_processed.csv")
    args = ap.parse_args()
    run(args.in_path, args.out_path)