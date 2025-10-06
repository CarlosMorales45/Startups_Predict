# src/data/preparar_datos.py
from pathlib import Path
import argparse
import pandas as pd
import numpy as np

from src.utils.helpers import (
    normalize_str, parse_money_to_float, parse_year, to_int_safe, map_yes_no
)

CAT_MAPS = {
    "sector": {
        "ai/ml": "AI/ML", "ai / ml": "AI/ML", "ai": "AI/ML", "ai-ml": "AI/ML",
        "logistics": "Logistics", "e-commerce": "E-commerce"
    },
    "estado_operativo": {
        "activo": "activo", "activo ": "activo", " en pausa": "en pausa",
        "cerrado": "cerrado", "prototipo": "prototipo"
    }
}

ORDERED_STATES = ["cerrado","en pausa","prototipo","activo"]  # para orden lógico

def load_raw(path):
    df = pd.read_csv(path, encoding="utf-8")
    return df

def clean_categoricals(df):
    for col in ["sector","ubicacion","estado_operativo"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: normalize_str(x, "lower_strip"))
            # map básicos
            if col in CAT_MAPS:
                df[col] = df[col].replace(CAT_MAPS[col])
            # estiliza a formato presentable
            if col == "sector":
                df[col] = df[col].str.replace("ai/ml","AI/ML", regex=False)
                df[col] = df[col].str.title().str.replace("Ai/Ml","AI/ML")
            elif col == "ubicacion":
                df[col] = df[col].str.title()
            elif col == "estado_operativo":
                df[col] = df[col].replace({"Activo":"activo","En Pausa":"en pausa"})
    # categoriza ordenado
    if "estado_operativo" in df.columns:
        df["estado_operativo"] = pd.Categorical(df["estado_operativo"], 
                                                categories=ORDERED_STATES, ordered=True)
    return df

def clean_numeric_and_flags(df):
    # Asegura índice limpio tras operaciones
    df = df.reset_index(drop=True)
    # monto_financiado
    if "monto_financiado" in df.columns:
        df["monto_financiado"] = df["monto_financiado"].apply(parse_money_to_float)

    # num_rondas
    if "num_rondas" in df.columns:
        df["num_rondas"] = df["num_rondas"].apply(to_int_safe)

    # inversores_destacados
    if "inversores_destacados" in df.columns:
        df["inversores_destacados"] = df["inversores_destacados"].apply(map_yes_no)

    # tiempo_fundacion
    if "tiempo_fundacion" in df.columns:
        df["tiempo_fundacion"] = df["tiempo_fundacion"].apply(parse_year).apply(to_int_safe)

    # presencia_redes -> mantener “suciedad” menor, pero recortar outliers extremos
    if "presencia_redes" in df.columns:
        df["presencia_redes"] = pd.to_numeric(df["presencia_redes"], errors="coerce")
        df["presencia_redes"] = df["presencia_redes"].clip(lower=-5, upper=150)

    # Flags de ausencia
    for col in ["monto_financiado","num_rondas","tamano_equipo","exp_fundadores",
                "presencia_redes","ubicacion","inversores_destacados","tiempo_fundacion"]:
        if col in df.columns:
            df[f"flag_na_{col}"] = df[col].isna().astype(int)
    return df

def basic_impute(df):
    """Imputación ligera SOLO si quieres salir del paso.
    Si prefieres imputar en el notebook o en un Pipeline de sklearn, puedes omitir esta función."""
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object","category"]).columns

    df[num_cols] = df[num_cols].apply(lambda s: s.fillna(s.median()))
    df[cat_cols] = df[cat_cols].apply(lambda s: s.fillna("desconocido"))
    return df

def prepare_dataframe(df, do_impute=False):
    df = clean_categoricals(df)
    df = clean_numeric_and_flags(df)
    if do_impute:
        df = basic_impute(df)
    return df

def run(in_path, out_path=None, impute=False):
    df_raw = load_raw(in_path)
    df_clean = prepare_dataframe(df_raw, do_impute=impute)
    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.suffix.lower() == ".parquet":
            df_clean.to_parquet(out_path, index=False)
        else:
            df_clean.to_csv(out_path, index=False, encoding="utf-8")
        print(f"✅ Datos limpios guardados en: {out_path}")
        print(f"Shape final: {df_clean.shape}")
    return df_clean

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", dest="out_path", default="data/processed/clean_base.parquet")
    ap.add_argument("--impute", action="store_true", help="aplica imputación básica")
    args = ap.parse_args()
    run(args.in_path, args.out_path, impute=args.impute)