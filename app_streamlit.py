import sys
from pathlib import Path

import streamlit as st

# --------------------------------------------------
# Localizar raíz del proyecto y ajustar sys.path
# --------------------------------------------------
THIS_FILE = Path(__file__).resolve()
THIS_DIR = THIS_FILE.parent

# Buscamos un directorio que tenga 'src' y (opcionalmente) 'data'
PROJECT_ROOT = None
for cand in [THIS_DIR, *THIS_DIR.parents]:
    if (cand / "src").exists():
        PROJECT_ROOT = cand
        break

if PROJECT_ROOT is None:
    PROJECT_ROOT = THIS_DIR  # fallback

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------
# Configuración de página
# --------------------------------------------------
st.set_page_config(
    page_title="Asistente de viabilidad de startups",
    page_icon="💡",
    layout="centered",
)

st.write(f"🛠️ Proyecto cargado desde: `{PROJECT_ROOT}`")

# --------------------------------------------------
# Import del predictor con manejo de errores
# --------------------------------------------------
try:
    from src.models.predictor import predecir_viabilidad
except Exception as e:
    st.error("❌ No se pudo importar `predecir_viabilidad` desde `src.models.predictor`.")
    st.exception(e)
    st.stop()

st.title("💡 Asistente de viabilidad de startups")
st.write(
    """
Describe tu startup y, si quieres, agrega algunos datos numéricos.
  
Si no sabes algún valor (monto, número de rondas, etc.), **déjalo vacío**.  
El sistema usará solo la información disponible.
"""
)

# -----------------------------
# Formulario principal
# -----------------------------
with st.form("form_startup"):

    st.subheader("1. Descripción (obligatoria)")
    descripcion = st.text_area(
        "Cuenta brevemente tu startup:",
        height=180,
        placeholder=(
            "Ejemplo: Plataforma SaaS de facturación electrónica para pymes, "
            "con 300 clientes activos, MRR de 25k USD y crecimiento mensual del 20%..."
        ),
    )

    st.subheader("2. Datos tabulares (opcionales)")
    st.caption("Puedes dejar en blanco lo que no sepas. El modelo tabular solo se usará si hay información suficiente.")

    col1, col2 = st.columns(2)

    # --- Columna izquierda ---
    with col1:
        monto_financiado_str = st.text_input(
            "Monto financiado aprox. (USD)",
            placeholder="Ej: 500000, 25k, 1.2M...",
        )
        num_rondas_str = st.text_input(
            "Número de rondas de inversión",
            placeholder="Ej: 0, 1, 2...",
        )
        tamano_equipo_str = st.text_input(
            "Tamaño del equipo",
            placeholder="Ej: 3, 10, 25...",
        )
        exp_fundadores_str = st.text_input(
            "Años de experiencia acumulada de fundadores",
            placeholder="Ej: 5, 10...",
        )

    # --- Columna derecha ---
    with col2:
        presencia_redes_str = st.text_input(
            "Presencia en redes (0–100)",
            placeholder="Ej: 40, 80...",
        )

        inversores_destacados = st.selectbox(
            "¿Inversores destacados?",
            options=["", "si", "no"],
            format_func=lambda x: "— (no lo sé / prefiero no decir)" if x == "" else x,
        )

        tiempo_fundacion_str = st.text_input(
            "Año de fundación",
            placeholder="Ej: 2018...",
        )

        sector = st.selectbox(
            "Sector principal",
            options=["", "Healthtech", "E-commerce", "Logistics", "AI/ML", "Fintech", "Edtech"],
            format_func=lambda x: "— (no especificar)" if x == "" else x,
        )

        ubicacion = st.text_input(
            "Ubicación principal (ciudad/país)",
            placeholder="Ej: Lima, Perú",
        )

        estado_operativo = st.selectbox(
            "Estado operativo",
            options=["", "activo", "en pausa", "prototipo", "cerrado"],
            format_func=lambda x: "— (no especificar)" if x == "" else x,
        )

    st.subheader("3. Configuración de la combinación")
    alpha = st.slider(
        "Peso del modelo tabular (α)",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="α=0 usa solo texto, α=1 usa solo el modelo tabular (si hay datos).",
    )

    submitted = st.form_submit_button("Evaluar viabilidad")

# -----------------------------
# Helper para castear (campos opcionales)
# -----------------------------
def _none_if_empty(s: str):
    s = str(s).strip()
    return s if s != "" else None


# -----------------------------
# Lógica de predicción
# -----------------------------
if submitted:
    if not descripcion or not descripcion.strip():
        st.error("Por favor, escribe al menos una descripción de la startup.")
    else:
        raw_inputs = {
            "descripcion": descripcion,
            "monto_financiado": _none_if_empty(monto_financiado_str),
            "num_rondas": _none_if_empty(num_rondas_str),
            "tamano_equipo": _none_if_empty(tamano_equipo_str),
            "exp_fundadores": _none_if_empty(exp_fundadores_str),
            "presencia_redes": _none_if_empty(presencia_redes_str),
            "inversores_destacados": _none_if_empty(inversores_destacados),
            "tiempo_fundacion": _none_if_empty(tiempo_fundacion_str),
            "sector": _none_if_empty(sector),
            "ubicacion": _none_if_empty(ubicacion),
            "estado_operativo": _none_if_empty(estado_operativo),
        }

        try:
            res = predecir_viabilidad(raw_inputs, alpha=alpha)
        except Exception as e:
            st.error("Error al calcular la predicción.")
            st.exception(e)
        else:
            st.markdown("---")
            st.subheader("Resultado")

            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric(
                    "Probabilidad (texto)",
                    f"{res['proba_texto']:.2f}",
                    help=f"Umbral texto demo: {res['thr_texto']:.2f}",
                )
                st.write(f"Predicción texto: **{'Viable' if res['pred_texto'] else 'No viable'}**")

            with col_b:
                if res["proba_tabular"] is not None:
                    st.metric(
                        "Probabilidad (tabular)",
                        f"{res['proba_tabular']:.2f}",
                    )
                    st.write(f"Predicción tabular: **{'Viable' if res['pred_tabular'] else 'No viable'}**")
                else:
                    st.write("Modelo tabular:")
                    st.info("No se usó (faltan datos o falló el preprocesado).")

            with col_c:
                st.metric(
                    "Probabilidad combinada",
                    f"{res['proba_combinada']:.2f}",
                )
                st.write(f"Predicción final: **{'Viable' if res['pred_combinada'] else 'No viable'}**")
                st.caption(f"α = {res['alpha']:.2f}  |  Usa tabular: {res['usa_tabular']}")

            st.markdown("### Explicación basada en el texto")
            st.write(res["explicacion_texto"]["interpretacion"])
            st.caption(res["explicacion_texto"]["justificacion"])

            with st.expander("Ver entrada enviada al modelo"):
                st.json(raw_inputs)
