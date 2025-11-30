# src/explainability/explicacion.py
from __future__ import annotations
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42

# ---------------- TABULAR ----------------
def permutation_importance_top(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    top: int = 15,
    scoring: str = "roc_auc",
    n_repeats: int = 10,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Importancia por permutación:
    importance = métrica_original - métrica_permutada (promedio n_repeats).
    """
    r = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=random_state, scoring=scoring
    )
    df_imp = (
        pd.DataFrame({"feature": X.columns, "importance": r.importances_mean})
        .sort_values("importance", ascending=False)
        .head(top)
        .reset_index(drop=True)
    )
    return df_imp


# ---------------- TEXTO ----------------
DEFAULT_STOP_ES = {
    "a","ante","bajo","con","contra","de","desde","durante","en","entre","hacia",
    "hasta","para","por","según","sin","sobre","tras","el","la","los","las","un",
    "una","unos","unas","lo","al","del","y","o","u","e","que","se","su","sus",
    "es","son","ser","fue","han","ha","como","más","menos","muy","ya","no","sí",
    "si","pero","también","porque","cuando","este","esta","estos","estas"
}

def _unwrap_estimator(obj):
    """Si viene un CV wrapper (GridSearchCV/RandomizedSearchCV), devuelve best_estimator_."""
    if hasattr(obj, "best_estimator_"):
        return obj.best_estimator_
    return obj

def _find_tfidf_and_clf(pipe) -> Tuple[TfidfVectorizer, Any]:
    """
    Busca robustamente TF-IDF y clasificador dentro de un Pipeline.
    No depende del nombre de steps.
    """
    pipe = _unwrap_estimator(pipe)

    vec = None
    clf = None

    # Caso Pipeline normal
    if isinstance(pipe, Pipeline):
        for _, step in pipe.steps:
            if vec is None and isinstance(step, TfidfVectorizer):
                vec = step
            if clf is None and (isinstance(step, LogisticRegression) or hasattr(step, "coef_")):
                clf = step

    # Fallback por si el objeto tiene named_steps pero no es Pipeline estricto
    if vec is None and hasattr(pipe, "named_steps"):
        for step in pipe.named_steps.values():
            if isinstance(step, TfidfVectorizer):
                vec = step
                break

    if clf is None and hasattr(pipe, "named_steps"):
        for step in pipe.named_steps.values():
            if isinstance(step, LogisticRegression) or hasattr(step, "coef_"):
                clf = step
                break

    if vec is None or clf is None:
        raise ValueError(
            "No se encontró TF-IDF o clasificador en el pipeline. "
            "Verifica que `modelo_texto.joblib` sea un Pipeline TF-IDF + LR "
            "(no solo el clasificador)."
        )

    return vec, clf

def _top_terms_from_text(pipe, descripcion: str, stopwords: set = DEFAULT_STOP_ES) -> List[Tuple[str, float]]:
    """Devuelve términos con contribución positiva (TF-IDF × coef)."""
    vec, clf = _find_tfidf_and_clf(pipe)
    X1 = vec.transform([descripcion])

    contrib = X1.multiply(clf.coef_)
    contrib = np.asarray(contrib.todense()).ravel()
    feats = vec.get_feature_names_out()

    out = []
    for i, c in enumerate(contrib):
        if c <= 0:
            continue
        t = feats[i]
        if len(t) < 3 or t in stopwords:
            continue
        out.append((t, float(c)))

    out.sort(key=lambda x: x[1], reverse=True)
    return out

def _translate_tokens(tokens: List[str]) -> List[str]:
    """Mini mapeo a lenguaje más humano."""
    mapping = {
        # tokens originales
        "mrr": "ingresos recurrentes (MRR)",
        "clientes": "clientes activos",
        "usuarios": "usuarios activos",
        "b2b": "acuerdos B2B",
        "semilla": "financiamiento semilla",
        "ronda": "ronda de inversión",
        "crecimiento": "crecimiento sostenido",
        "rentable": "rentabilidad",
        "mvp": "MVP listo",
        "beta": "beta validada",
        "traccion": "tracción en mercado",
        "equipo": "equipo fundador",
        "experiencia": "experiencia del equipo",
        "años": "años de experiencia",
        # nuevos tags numéricos
        "__monto_pequeno__": "monto de inversión pequeño",
        "__monto_medio__": "monto de inversión medio",
        "__monto_alto__": "monto de inversión alto",
        "__monto_muy_alto__": "monto de inversión muy alto",
        "__clientes_bajos__": "pocos clientes o usuarios",
        "__clientes_medios__": "base de clientes en crecimiento",
        "__clientes_altos__": "base de clientes grande",
        "__muchos_clientes__": "base de clientes muy grande",
        "__crecimiento_bajo__": "crecimiento bajo en porcentajes",
        "__crecimiento_medio__": "crecimiento moderado",
        "__crecimiento_alto__": "crecimiento alto",
        "__crecimiento_explosivo__": "crecimiento explosivo",
        "__tiene_mrr__": "ingresos recurrentes (MRR)",
    }
    return [mapping.get(t, t) for t in tokens]

def _spanish_join(items: List[str]) -> str:
    items = [i for i in items if i]
    if len(items) == 0: return ""
    if len(items) == 1: return items[0]
    if len(items) == 2: return f"{items[0]} y {items[1]}"
    return ", ".join(items[:-1]) + f" y {items[-1]}"

def explicar_texto(
    pipe,
    descripcion: str,
    stopwords: set = DEFAULT_STOP_ES,
    topk: int = 5,
    thr: float | None = None
) -> Dict[str, Any]:
    """
    Explicación entendible para público general.

    Si `thr` es None, intenta usar `pipe._thr_demo` (umbral calibrado en entrenamiento).
    Si no existe, usa 0.5.
    """
    pipe = _unwrap_estimator(pipe)

    if thr is None:
        thr = float(getattr(pipe, "_thr_demo", 0.5))

    prob = float(pipe.predict_proba([descripcion])[0, 1])
    viable = bool(prob >= thr)

    top_terms = _top_terms_from_text(pipe, descripcion, stopwords=stopwords)[:topk]
    just = "; ".join([f"{t}:{w:.3f}" for t, w in top_terms]) if top_terms else "(sin señales específicas útiles)"

    tokens_legibles = _translate_tokens([t for t, _ in top_terms[:3]])
    motivos_legibles = _spanish_join(tokens_legibles)

    if viable:
        frase = (
            f"La startup **sí parece viable** (probabilidad {prob:.2f}) "
            f"porque menciona señales positivas como {motivos_legibles or 'tracción, mercado y experiencia del equipo'}."
        )
    else:
        frase = (
            f"La startup **no parece viable** (probabilidad {prob:.2f}) "
            f"porque no aparecen suficientes señales de tracción o madurez; lo más relevante fue "
            f"{motivos_legibles or 'poca evidencia clara en el texto'}."
        )

    return {
        "probabilidad": prob,
        "umbral": float(thr),
        "viable": viable,
        "justificacion": f"Aportes: {just}",
        "interpretacion": frase
    }
