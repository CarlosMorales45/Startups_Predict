# src/features/text_features.py
from __future__ import annotations
from typing import List
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class KeywordFeatures(BaseEstimator, TransformerMixin):
    """
    Features simples por keywords y patrones numéricos en descripciones.

    Salida por fila (n_samples, 8):
      0. conteo de señales positivas (tracción, inversión, crecimiento)
      1. conteo de señales de riesgo (fase temprana, sin ventas, cierre)
      2. número de menciones de montos de dinero
      3. log1p del mayor monto detectado (en unidades aproximadas)
      4. número de menciones de clientes/usuarios/descargas
      5. log1p del mayor número de clientes/usuarios detectado
      6. número de porcentajes detectados
      7. valor máximo de porcentaje detectado
    """
    def __init__(
        self,
        positive_keywords: List[str] | None = None,
        risk_keywords: List[str] | None = None,
    ):
        self.positive_keywords = positive_keywords or [
            "mrr","ingresos","recurrente","clientes","ventas","contrato","acuerdo","b2b",
            "crecimiento","rentable","profit","flujo","usuarios activos","retención",
            "semilla","inversion","financiamiento","ronda"
        ]
        self.risk_keywords = risk_keywords or [
            "sin ventas","sin clientes","deuda","pérdidas","cerrado","quiebra","abandono",
            "prototipo","idea","mvp","beta","sin tracción","problemas legales"
        ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = [str(x).lower() for x in X]

        money_pattern = re.compile(
            r"(\d+[.,]?\d*)\s*(k|mil|m|mm|millones)?\s*(usd|us\$|\$)?",
            flags=re.IGNORECASE,
        )
        clients_pattern = re.compile(
            r"(\d+[.,]?\d*)\s+(clientes?|usuarios?|empresas|negocios|tiendas|descargas|suscriptores)",
            flags=re.IGNORECASE,
        )
        pct_pattern = re.compile(r"(\d+[.,]?\d*)\s*%", flags=re.IGNORECASE)

        rows = []
        for s in X:
            # keywords
            pos = sum(1 for k in self.positive_keywords if re.search(rf"\b{re.escape(k)}\b", s))
            risk = sum(1 for k in self.risk_keywords if re.search(rf"\b{re.escape(k)}\b", s))

            # money
            money_vals = []
            for m in money_pattern.finditer(s):
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
                val = base * factor
                if val > 0:
                    money_vals.append(val)
            money_mentions = len(money_vals)
            max_money = max(money_vals) if money_vals else 0.0

            # clients / usuarios
            client_vals = []
            for m in clients_pattern.finditer(s):
                num_str, _ = m.groups()
                clean = num_str.replace(".", "").replace(",", "")
                try:
                    n = float(clean)
                except ValueError:
                    continue
                if n > 0:
                    client_vals.append(n)
            client_mentions = len(client_vals)
            max_clients = max(client_vals) if client_vals else 0.0

            # porcentajes
            pct_vals = []
            for m in pct_pattern.finditer(s):
                num_str = m.group(1)
                clean = num_str.replace(",", ".")
                try:
                    p = float(clean)
                except ValueError:
                    continue
                if p > 0:
                    pct_vals.append(p)
            pct_mentions = len(pct_vals)
            max_pct = max(pct_vals) if pct_vals else 0.0

            row = [
                float(pos),
                float(risk),
                float(money_mentions),
                float(np.log1p(max_money)),
                float(client_mentions),
                float(np.log1p(max_clients)),
                float(pct_mentions),
                float(max_pct),
            ]
            rows.append(row)

        return np.asarray(rows, dtype=float)
