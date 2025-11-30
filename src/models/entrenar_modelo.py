# src/models/entrenar_modelo.py
from __future__ import annotations
from pathlib import Path
import argparse, json, re
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from scipy.stats import randint as sp_randint, uniform
import joblib

RANDOM_STATE = 42


def _augment_text_with_numeric_signals(text: str) -> str:
    """Enriquece la descripción con tags sobre montos, clientes, porcentajes
    y algunas señales textuales de riesgo.

    No eliminamos nada del texto original; solo añadimos tokens especiales
    como "__monto_alto__" o "__sin_clientes__" que TF-IDF podrá usar.
    """
    if not isinstance(text, str):
        text = str(text)
    raw = text.lower()
    tags: list[str] = []

    # -------------------- MONTOS --------------------
    money_pattern = re.compile(
        r"(\d+[.,]?\d*)\s*(k|mil|m|mm|millones)?\s*(usd|us\$|\$)?",
        flags=re.IGNORECASE,
    )
    money_vals = []
    for m in money_pattern.finditer(raw):
        num_str, mult_str, _ = m.groups()
        clean = num_str.replace(".", "").replace(",", "")
        try:
            base = float(clean)
        except ValueError:
            continue
        factor = 1.0
        if mult_str:
            ms = mult_str.lower()
            if ms in {"k", "mil"}:
                factor = 1e3
            elif ms in {"m", "mm", "millones"}:
                factor = 1e6
        money = base * factor
        if money > 0:
            money_vals.append(money)

    if money_vals:
        max_money = max(money_vals)
        if max_money >= 500_000:
            tags.append("__monto_muy_alto__")
        elif max_money >= 200_000:
            tags.append("__monto_alto__")
        elif max_money >= 50_000:
            tags.append("__monto_medio__")
        else:
            tags.append("__monto_pequeno__")

    # -------------------- CLIENTES / USUARIOS --------------------
    clients_pattern = re.compile(
        r"(\d+[.,]?\d*)\s+(clientes?|usuarios?|empresas|negocios|tiendas|descargas|suscriptores)",
        flags=re.IGNORECASE,
    )
    client_vals = []
    for m in clients_pattern.finditer(raw):
        num_str, _ = m.groups()
        clean = num_str.replace(".", "").replace(",", "")
        try:
            n = float(clean)
        except ValueError:
            continue
        if n > 0:
            client_vals.append(n)

    if client_vals:
        max_clients = max(client_vals)
        if max_clients >= 5_000:
            tags.append("__muchos_clientes__")
        elif max_clients >= 1_000:
            tags.append("__clientes_altos__")
        elif max_clients >= 200:
            tags.append("__clientes_medios__")
        else:
            tags.append("__clientes_bajos__")

    # -------------------- PORCENTAJES --------------------
    pct_pattern = re.compile(r"(\d+[.,]?\d*)\s*%", flags=re.IGNORECASE)
    pct_vals = []
    for m in pct_pattern.finditer(raw):
        num_str = m.group(1)
        clean = num_str.replace(",", ".")
        try:
            p = float(clean)
        except ValueError:
            continue
        if p > 0:
            pct_vals.append(p)

    if pct_vals:
        max_pct = max(pct_vals)
        if max_pct >= 100:
            tags.append("__crecimiento_explosivo__")
        elif max_pct >= 50:
            tags.append("__crecimiento_alto__")
        elif max_pct >= 20:
            tags.append("__crecimiento_medio__")
        else:
            tags.append("__crecimiento_bajo__")

    # -------------------- SEÑALES TEXTUALES DE RIESGO --------------------
    # Esto ayuda a que frases muy típicas de proyectos poco maduros empujen la probabilidad hacia "no viable".
    if "sin clientes" in raw or "sin usuarios" in raw:
        tags.append("__sin_clientes__")
    if "sin ventas" in raw or "sin ingresos" in raw:
        tags.append("__sin_ingresos__")
    if "solo una idea" in raw or "fase conceptual" in raw or "fase idea" in raw:
        tags.append("__fase_idea__")
    if "prototipo" in raw and "sin pilotos" in raw:
        tags.append("__prototipo_sin_validar__")
    if "deuda" in raw or "deudas" in raw or "pérdidas" in raw or "perdidas" in raw:
        tags.append("__problemas_financieros__")
    if "cerrado" in raw or "cerrada" in raw:
        tags.append("__estado_cerrado__")

    # -------------------- MRR --------------------
    if "mrr" in raw or "ingresos recurrentes" in raw:
        tags.append("__tiene_mrr__")

    if not tags:
        return text

    extra = " " + " ".join(sorted(set(tags)))
    return text + extra


def _find_project_root(start: Path | None = None) -> Path:
    p = start or Path.cwd()
    for cand in [p, *p.parents]:
        if (cand / "data" / "processed").exists():
            return cand
    raise FileNotFoundError("No se encontró la carpeta 'data/processed'.")


def _split_70_15_15(X, y):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_valid, X_test, y_valid, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def _eval_model(clf, Xv, yv, name: str):
    proba = clf.predict_proba(Xv)[:, 1] if hasattr(clf, "predict_proba") else clf.decision_function(Xv)
    auc = roc_auc_score(yv, proba)
    pred = (proba >= 0.5).astype(int)
    return dict(
        model=name,
        auc=float(auc),
        f1=float(f1_score(yv, pred)),
        acc=float(accuracy_score(yv, pred)),
        prec=float(precision_score(yv, pred, zero_division=0)),
        rec=float(recall_score(yv, pred, zero_division=0)),
    )


def _calibrar_umbral_texto(proba_valid: np.ndarray, y_valid: pd.Series, prefer_prec: float = 0.8):
    """
    Busca umbrales razonables para el modelo de texto.

    - Recorre thresholds entre 0.10 y 0.90.
    - Calcula F1, precision y recall en el conjunto de validación.
    - thr_f1  : umbral que maximiza F1.
    - thr_demo: umbral más alto con precision >= prefer_prec (por defecto 0.80).
               Si no hay ninguno, usa thr_f1.
    """
    ths = np.linspace(0.1, 0.9, 33)
    filas = []
    for t in ths:
        pred = (proba_valid >= t).astype(int)
        filas.append({
            "thr": float(t),
            "f1": float(f1_score(y_valid, pred)),
            "precision": float(precision_score(y_valid, pred, zero_division=0)),
            "recall": float(recall_score(y_valid, pred, zero_division=0)),
        })

    df_thr = pd.DataFrame(filas)
    thr_f1 = float(df_thr.loc[df_thr["f1"].idxmax(), "thr"])

    candidatos = df_thr[df_thr["precision"] >= prefer_prec]
    if not candidatos.empty:
        # Usamos el umbral MÁS ESTRICTO que mantiene buena precisión
        thr_demo = float(candidatos.sort_values("thr").iloc[-1]["thr"])
    else:
        thr_demo = thr_f1

    return thr_f1, thr_demo, df_thr


def entrenar_tabular(in_csv: Path, out_model: Path, out_report: Path | None = None) -> dict:
    df = pd.read_csv(in_csv, encoding="utf-8")
    if "descripcion" in df.columns:
        df = df.drop(columns=["descripcion"])

    obj_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=False)

    y = df["viabilidad"].astype(int)
    X = df.drop(columns=["viabilidad"])

    X_train, X_valid, X_test, y_train, y_valid, y_test = _split_70_15_15(X, y)

    results = []

    dummy = DummyClassifier(strategy="stratified", random_state=RANDOM_STATE).fit(X_train, y_train)
    results.append(_eval_model(dummy, X_valid, y_valid, "Dummy"))

    logit = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
    ]).fit(X_train, y_train)
    results.append(_eval_model(logit, X_valid, y_valid, "LogReg"))

    tree = DecisionTreeClassifier(random_state=RANDOM_STATE).fit(X_train, y_train)
    results.append(_eval_model(tree, X_valid, y_valid, "DecisionTree"))

    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }
    rf_cv = GridSearchCV(rf, rf_grid, cv=5, scoring="roc_auc", n_jobs=-1).fit(X_train, y_train)
    best_rf = rf_cv.best_estimator_
    results.append(_eval_model(best_rf, X_valid, y_valid, f"RandomForest(best={rf_cv.best_params_})"))

    hgb = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    hgb_dist = {
        "max_depth": sp_randint(2, 16),
        "learning_rate": uniform(0.02, 0.28),
        "max_bins": sp_randint(32, 255),
    }
    hgb_cv = RandomizedSearchCV(
        hgb, hgb_dist, cv=5, n_iter=20, scoring="roc_auc",
        random_state=RANDOM_STATE, n_jobs=-1
    ).fit(X_train, y_train)
    best_hgb = hgb_cv.best_estimator_
    results.append(_eval_model(best_hgb, X_valid, y_valid, "HistGB(best)"))

    # Selección
    df_val = pd.DataFrame(results).sort_values("auc", ascending=False)
    top_name = df_val.iloc[0]["model"]
    if "HistGB" in top_name:
        final = best_hgb
    elif "RandomForest" in top_name:
        final = best_rf
    elif "LogReg" in top_name:
        final = logit
    elif "DecisionTree" in top_name:
        final = tree
    else:
        final = logit

    # Reentrenar en train+valid y evaluar
    X_trv = pd.concat([X_train, X_valid], axis=0)
    y_trv = pd.concat([y_train, y_valid], axis=0)
    final.fit(X_trv, y_trv)

    # <<< NUEVO: guardar columnas de features para el predictor.tabular >>>
    if hasattr(X_trv, "columns"):
        final._feature_names = X_trv.columns.tolist()
    # <<< FIN NUEVO >>>

    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final, out_model)
    print(f"✅ Modelo tabular guardado en: {out_model}")

    proba = final.predict_proba(X_test)[:, 1] if hasattr(final, "predict_proba") else final.decision_function(X_test)
    test = dict(
        auc=float(roc_auc_score(y_test, proba)),
        f1=float(f1_score(y_test, (proba >= 0.5).astype(int))),
        acc=float(accuracy_score(y_test, (proba >= 0.5).astype(int))),
    )
    report = {"valid_results": results, "selected": top_name, "test": test}
    if out_report:
        out_report.parent.mkdir(parents=True, exist_ok=True)
        out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"📝 Reporte de validación/test guardado en: {out_report}")

    return report


def entrenar_texto(in_csv: Path, out_model: Path, stopwords: set | None = None) -> dict:
    """
    Entrena el modelo SOLO con texto (descripcion).

    Cambios claves:
    - Split 70/15/15 (train/valid/test) para poder calibrar un umbral.
    - TF-IDF más permisivo (min_df=1, max_features=1500).
    - token_pattern acepta números/unidades (25k, 12%, 350k, usd, b2b).
    - strip_accents para unificar vocabulario.
    - class_weight='balanced' para reducir sesgo por desbalance.
    - Se guarda en el pipeline un atributo `_thr_demo` con el umbral recomendado.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import LogisticRegression

    STOP_ES = stopwords or {
        "a","ante","bajo","con","contra","de","desde","durante","en","entre","hacia",
        "hasta","para","por","según","sin","sobre","tras","el","la","los","las","un",
        "una","unos","unas","lo","al","del","y","o","u","e","que","se","su","sus",
        "es","son","ser","fue","han","ha","como","más","menos","muy","ya","no","sí",
        "si","pero","también","porque","cuando","este","esta","estos","estas"
    }

    df = pd.read_csv(in_csv, encoding="utf-8")
    assert {"descripcion", "viabilidad"}.issubset(df.columns), "Faltan columnas requeridas."
    X_text = df["descripcion"].astype(str)
    y = df["viabilidad"].astype(int)

    # Usamos el mismo esquema 70/15/15 que en el modelo tabular
    X_train, X_valid, X_test, y_train, y_valid, y_test = _split_70_15_15(X_text, y)

    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=1500,
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            stop_words=list(STOP_ES),
            token_pattern=r"(?u)\b[\wáéíóúñ%$]{2,}\b",
            sublinear_tf=True,
            strip_accents="unicode",
            preprocessor=_augment_text_with_numeric_signals,
        )),
        ("clf", LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced"
        ))
    ])

    # Entrenamos en train y calibramos umbral en valid
    pipe.fit(X_train, y_train)
    proba_valid = pipe.predict_proba(X_valid)[:, 1]

    thr_f1, thr_demo, df_thr = _calibrar_umbral_texto(proba_valid, y_valid, prefer_prec=0.8)

    # Re-entrenamos en train + valid con el umbral ya decidido
    X_trv = pd.concat([X_train.reset_index(drop=True), X_valid.reset_index(drop=True)], axis=0)
    y_trv = pd.concat([y_train.reset_index(drop=True), y_valid.reset_index(drop=True)], axis=0)
    pipe.fit(X_trv, y_trv)

    # Evaluación final en test (solo para tener referencia)
    proba_test = pipe.predict_proba(X_test)[:, 1]
    pred05 = (proba_test >= 0.5).astype(int)
    pred_f1 = (proba_test >= thr_f1).astype(int)
    pred_demo = (proba_test >= thr_demo).astype(int)

    report = {
        "auc": float(roc_auc_score(y_test, proba_test)),
        "f1@0.5": float(f1_score(y_test, pred05)),
        "thr_f1": float(thr_f1),
        "f1@thr_f1": float(f1_score(y_test, pred_f1)),
        "thr_demo": float(thr_demo),
        "f1@thr_demo": float(f1_score(y_test, pred_demo)),
        "precision@thr_demo": float(precision_score(y_test, pred_demo, zero_division=0)),
        "recall@thr_demo": float(recall_score(y_test, pred_demo, zero_division=0)),
    }

    # Guardamos el umbral dentro del propio pipeline para usarlo luego en la app / explicaciones
    pipe._thr_demo = float(thr_demo)
    pipe._thr_f1 = float(thr_f1)
    pipe._thr_grid = df_thr.to_dict(orient="list")  # opcional, por si quieres inspeccionarlo luego

    out_model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_model)
    print(f"✅ Modelo de texto guardado en: {out_model}")
    print(
        f"📌 Texto AUC_test={report['auc']:.3f} | "
        f"F1@0.5={report['f1@0.5']:.3f} | "
        f"thr_demo={report['thr_demo']:.2f} "
        f"(F1={report['f1@thr_demo']:.3f}, Prec={report['precision@thr_demo']:.3f}, "
        f"Rec={report['recall@thr_demo']:.3f})"
    )

    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tabular", "texto"], required=True, help="Entrenar modelo tabular o de texto")
    ap.add_argument("--in", dest="in_path", default=None, help="Ruta al CSV procesado")
    ap.add_argument("--out", dest="out_model", default=None, help="Ruta destino del .joblib")
    ap.add_argument("--report", dest="out_report", default=None, help="(solo tabular) JSON con métricas")
    args = ap.parse_args()

    root = _find_project_root()
    in_csv = Path(args.in_path) if args.in_path else (root / "data/processed/startups_sintetico_1000_processed.csv")
    out_model = Path(args.out_model) if args.out_model else (
        root / "models/modelo_tabular.joblib" if args.mode == "tabular" else root / "models/modelo_texto.joblib"
    )
    out_report = Path(args.out_report) if args.out_report else (root / "models/report_tabular.json")

    if args.mode == "tabular":
        entrenar_tabular(in_csv, out_model, out_report)
    else:
        entrenar_texto(in_csv, out_model)


if __name__ == "__main__":
    main()
