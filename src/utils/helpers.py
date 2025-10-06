# src/utils/helpers.py
import re
import numpy as np
import pandas as pd

def normalize_str(x, mode="lower_strip"):
    if pd.isna(x):
        return x
    s = str(x)
    if mode.startswith("lower"):
        s = s.lower()
    elif mode.startswith("title"):
        s = s.title()
    s = s.strip()
    return s

def parse_money_to_float(x):
    if pd.isna(x):
        return np.nan
    s = str(x)
    s = s.replace("USD", "").replace("$", "").strip()
    # reemplaza separadores comunes
    s = s.replace(".", "").replace(",", "")
    try:
        return float(s)
    except:
        # intenta extraer dígitos
        m = re.search(r"(\d+)", s)
        return float(m.group(1)) if m else np.nan

def parse_year(x):
    """Acepta 2018, '2018-aprox', '2014/2015', etc. Devuelve int (año) o NaN."""
    if pd.isna(x): 
        return np.nan
    s = str(x).strip()
    # '2014/2015' -> toma el primero
    m = re.search(r"(\d{4})", s)
    return int(m.group(1)) if m else np.nan

def to_int_safe(x):
    try:
        return int(float(x))
    except:
        return np.nan

def map_yes_no(x):
    if pd.isna(x): 
        return np.nan
    s = str(x).strip().lower()
    if s in {"si","sí","yes","y","true","1"}:
        return 1
    if s in {"no","false","0"}:
        return 0
    try:
        return int(float(s))
    except:
        return np.nan

if __name__ == "__main__":
    print("✅ helpers.py cargado correctamente.")
