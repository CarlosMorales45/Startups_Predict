# src/test/test_libraries.py
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import shap
import lime
import streamlit as st


def test_libraries():
    print("Pandas version:", pd.__version__)
    print("Numpy version:", np.__version__)
    print("Matplotlib version:", matplotlib.__version__)
    print("Seaborn version:", sns.__version__)

    # Carga un dataset de ejemplo de sklearn
    data = load_iris()
    X, y = data.data, data.target

    # Crear y entrenar un modelo rápido para test
    model = RandomForestClassifier()
    model.fit(X, y)
    print("Modelo RandomForest entrenado con datos iris.")

    # Test básico con SHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    print("SHAP calculado para las predicciones.")

    # Test básico con LIME (solo comprobación de importación)
    explainer_lime = lime.lime_tabular.LimeTabularExplainer(X, feature_names=data.feature_names, class_names=data.target_names, verbose=False, mode='classification')
    print("LIME inicializado correctamente.")

    # Test básico Streamlit (solo import)
    st.write("Streamlit importado correctamente.")


if __name__ == "__main__":
    test_libraries()