# src/models/evaluar_modelo.py
from __future__ import annotations
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix, classification_report, f1_score, precision_score, recall_score, accuracy_score

RANDOM_STATE = 42

def _find_project_root(start: Path | None = None) -> Path:
    p = start or Path.cwd()
    for cand in [p, *p.parents]:
        if (cand / "data" / "processed").exists():
            return cand
    raise FileNotFoundError("No se encontró la carpeta 'data/processed'.")

def evaluar_tabular(model_path: Path, in_csv: Path, out_dir: Path) -> dict:
    model = joblib.load(model_path)
    df = pd.read_csv(in_csv, encoding="utf-8")
    if "descripcion" in df.columns:
        df = df.drop(columns=["descripcion"])
    obj_cols = df.select_dtypes(include=["object","category"]).columns.tolist()
    if obj_cols:
        df = pd.get_dummies(df, columns=obj_cols, drop_first=False)
    y = df["viabilidad"].astype(int)
    X = df.drop(columns=["viabilidad"])

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    proba = model.predict_proba(X_test)[:,1] if hasattr(model,"predict_proba") else model.decision_function(X_test)
    fpr, tpr, _ = roc_curve(y_test, proba)
    pr, rc, _ = precision_recall_curve(y_test, proba)

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(); plt.plot(fpr, tpr); plt.title("ROC"); plt.xlabel("FPR"); plt.ylabel("TPR")
    roc_png = out_dir/"roc_tabular.png"; plt.savefig(roc_png, bbox_inches="tight"); plt.close()

    plt.figure(); plt.plot(rc, pr); plt.title("PR"); plt.xlabel("Recall"); plt.ylabel("Precision")
    pr_png = out_dir/"pr_tabular.png"; plt.savefig(pr_png, bbox_inches="tight"); plt.close()

    pred05 = (proba >= 0.5).astype(int)
    ths = np.linspace(0.1, 0.9, 17)
    scores = []
    for t in ths:
        pred = (proba >= t).astype(int)
        scores.append((float(t), float(f1_score(y_test, pred))))
    best_t, best_f1 = max(scores, key=lambda x: x[1])
    pred_best = (proba >= best_t).astype(int)

    report = {
        "auc": float(roc_auc_score(y_test, proba)),
        "f1@0.5": float(f1_score(y_test, pred05)),
        "f1@best": float(best_f1),
        "best_threshold": float(best_t),
        "acc@0.5": float(accuracy_score(y_test, pred05)),
        "precision@0.5": float(precision_score(y_test, pred05, zero_division=0)),
        "recall@0.5": float(recall_score(y_test, pred05, zero_division=0)),
        "cm@0.5": confusion_matrix(y_test, pred05).tolist(),
        "cm@best": confusion_matrix(y_test, pred_best).tolist(),
        "roc_png": str(roc_png),
        "pr_png": str(pr_png)
    }
    (out_dir/"report_tabular_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Evaluación tabular guardada en: {out_dir}")
    return report

def evaluar_texto(model_path: Path, in_csv: Path, out_dir: Path) -> dict:
    model = joblib.load(model_path)
    df = pd.read_csv(in_csv, encoding="utf-8")
    X = df["descripcion"].astype(str)
    y = df["viabilidad"].astype(int)

    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    proba = model.predict_proba(X_test)[:,1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    pr, rc, _ = precision_recall_curve(y_test, proba)

    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(); plt.plot(fpr, tpr); plt.title("ROC (texto)"); plt.xlabel("FPR"); plt.ylabel("TPR")
    roc_png = out_dir/"roc_texto.png"; plt.savefig(roc_png, bbox_inches="tight"); plt.close()

    plt.figure(); plt.plot(rc, pr); plt.title("PR (texto)"); plt.xlabel("Recall"); plt.ylabel("Precision")
    pr_png = out_dir/"pr_texto.png"; plt.savefig(pr_png, bbox_inches="tight"); plt.close()

    # Métricas a distintos umbrales
    pred05 = (proba >= 0.5).astype(int)

    ths = np.linspace(0.1, 0.9, 17)
    scores = []
    for t in ths:
        pred = (proba >= t).astype(int)
        scores.append((float(t), float(f1_score(y_test, pred))))
    best_t, best_f1 = max(scores, key=lambda x: x[1])
    pred_best = (proba >= best_t).astype(int)

    # Umbral recomendado en entrenamiento (si existe)
    thr_demo = float(getattr(model, "_thr_demo", 0.5))
    pred_demo = (proba >= thr_demo).astype(int)

    report = {
        "auc": float(roc_auc_score(y_test, proba)),
        "f1@0.5": float(f1_score(y_test, pred05)),
        "f1@best": float(best_f1),
        "best_threshold": float(best_t),
        "thr_demo": float(thr_demo),
        "f1@demo": float(f1_score(y_test, pred_demo)),
        "acc@0.5": float(accuracy_score(y_test, pred05)),
        "acc@demo": float(accuracy_score(y_test, pred_demo)),
        "precision@0.5": float(precision_score(y_test, pred05, zero_division=0)),
        "precision@demo": float(precision_score(y_test, pred_demo, zero_division=0)),
        "recall@0.5": float(recall_score(y_test, pred05, zero_division=0)),
        "recall@demo": float(recall_score(y_test, pred_demo, zero_division=0)),
        "cm@0.5": confusion_matrix(y_test, pred05).tolist(),
        "cm@demo": confusion_matrix(y_test, pred_demo).tolist(),
        "cm@best": confusion_matrix(y_test, pred_best).tolist(),
        "roc_png": str(roc_png),
        "pr_png": str(pr_png)
    }
    (out_dir/"report_texto_eval.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"✅ Evaluación de texto guardada en: {out_dir}")
    return report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["tabular","texto"], required=True, help="Evaluar modelo tabular o de texto")
    ap.add_argument("--model", dest="model_path", required=True, help="Ruta al .joblib a evaluar")
    ap.add_argument("--in", dest="in_path", default=None, help="Ruta al CSV procesado")
    ap.add_argument("--outdir", dest="out_dir", default=None, help="Carpeta de salida para gráficos y JSON")
    args = ap.parse_args()

    root = _find_project_root()
    in_csv = Path(args.in_path) if args.in_path else (root/"data/processed/startups_sintetico_1000_processed.csv")
    out_dir = Path(args.out_dir) if args.out_dir else (root/"models/eval")

    if args.mode == "tabular":
        evaluar_tabular(Path(args.model_path), in_csv, out_dir)
    else:
        evaluar_texto(Path(args.model_path), in_csv, out_dir)

if __name__ == "__main__":
    main()
