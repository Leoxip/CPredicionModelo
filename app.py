import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

# ==========================
# CONFIG STREAMLIT
# ==========================
st.set_page_config(
    page_title="Riesgo de Preeclampsia",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Predicción de Riesgo de Preeclampsia")
st.caption("Aplicación de apoyo clínico — no reemplaza evaluación médica.")

# ==========================
# INTENTAR IMPORTAR SHAP
# ==========================
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

# ==========================
# CARGA DE ARTEFACTOS
# ==========================
ART_DIR = os.path.join("artefactos", "v1")


@st.cache_resource
def load_artifacts():
    # Cargar JSON
    with open(os.path.join(ART_DIR, "input_schema.json"), "r", encoding="utf-8") as f:
        input_schema = json.load(f)
    with open(os.path.join(ART_DIR, "label_map.json"), "r", encoding="utf-8") as f:
        label_map = json.load(f)
    with open(os.path.join(ART_DIR, "decision_policy.json"), "r", encoding="utf-8") as f:
        policy = json.load(f)

    winner = policy["winner"]
    threshold = float(policy.get("threshold", 0.5))

    # Pipeline ganador
    pipe_path = os.path.join(ART_DIR, f"pipeline_{winner}.joblib")
    pipe = joblib.load(pipe_path)

    # Datos de ejemplo para dashboard / SHAP
    sample_df = None
    sample_path = os.path.join(ART_DIR, "sample_inputs.json")
    if os.path.exists(sample_path):
        with open(sample_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            raw = [raw]
        sample_df = pd.DataFrame(raw)

    rev_label = {v: k for k, v in label_map.items()}
    features = list(input_schema.keys())

    return pipe, input_schema, label_map, rev_label, threshold, features, policy, sample_df


PIPE, INPUT_SCHEMA, LABEL_MAP, REV_LABEL, THRESHOLD, FEATURES, POLICY, SAMPLE_DF = load_artifacts()

TEST_METRICS = POLICY.get("test_metrics", {})
# Estos pueden no existir, por eso usamos get con default []
Y_TRUE = np.array(POLICY.get("y_true", []))
Y_PROBA = np.array(POLICY.get("y_pred_proba", []))

# ==========================
# FUNCIONES AUXILIARES
# ==========================
def _coerce_and_align(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Asegura que todas las columnas del schema existan.
    - Convierte numéricas a número.
    - Deja categóricas como STRING (ej. 'SI'/'NO') para que coincida con el entrenamiento.
    """
    for col, typ in INPUT_SCHEMA.items():
        if col not in df.columns:
            df[col] = np.nan

        t = str(typ).lower()
        if t.startswith("int") or t.startswith("float"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            # IMPORTANTE: mantener texto 'SI'/'NO' tal cual
            df[col] = df[col].astype("string")

    # Asegurar orden correcto
    return df[FEATURES]


def predict_batch(records, thr: float | None = None):
    thr = THRESHOLD if thr is None else float(thr)

    if isinstance(records, dict):
        records = [records]

    df = pd.DataFrame(records)
    df = _coerce_and_align(df)

    # Aquí usamos el pipeline EXACTAMENTE como fue entrenado
    proba = PIPE.predict_proba(df)[:, 1]
    preds = (proba >= thr).astype(int)

    results = []
    for p, y in zip(proba, preds):
        results.append(
            {
                "proba": float(p),
                "pred_int": int(y),
                "pred_label": REV_LABEL[int(y)],
                "threshold": thr,
            }
        )
    return results


def get_background_for_shap(max_samples: int = 100):
    if SAMPLE_DF is None or SAMPLE_DF.empty:
        return None

    df = SAMPLE_DF.copy()
    df = _coerce_and_align(df)

    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)

    return df


def style_box(text: str, color: str):
    st.markdown(
        f"""
        <div style="padding:15px;border-radius:10px;background:{color}20;
                    border:1px solid {color}66;margin-bottom:15px;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.header("ℹ️ Modelo")
    st.markdown(f"**Modelo ganador:** `{POLICY.get('winner', 'N/A')}`")
    st.markdown(f"**Umbral:** `{THRESHOLD:.2f}`")

    st.markdown("### Métricas en test")
    if TEST_METRICS:
        for k, v in TEST_METRICS.items():
            if isinstance(v, (int, float)):
                st.metric(k.upper(), f"{v:.3f}")
    else:
        st.info("No hay métricas en decision_policy.json")

    st.markdown("---")
    st.markdown(f"SHAP disponible: {'✅' if HAS_SHAP else '❌'}")

# ==========================
# TABS
# ==========================
tab_pred, tab_model, tab_shap, tab_dash = st.tabs(
    ["🔮 Predicción", "📊 Análisis del modelo", "🧠 Interpretabilidad (SHAP)", "📈 Dashboard"]
)

# ==========================
# TAB 1: PREDICCIÓN
# ==========================
with tab_pred:
    st.subheader("📋 Datos de la paciente")

    with st.form("form_paciente"):
        c1, c2 = st.columns(2)

        with c1:
            edad = st.number_input("Edad (años)", min_value=10, max_value=60, value=30)
            imc = st.number_input("IMC", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
            p_sis = st.number_input("Presión sistólica (mmHg)", min_value=70, max_value=250, value=120)
            p_dia = st.number_input("Presión diastólica (mmHg)", min_value=40, max_value=150, value=80)

        with c2:
            hipertension = st.selectbox("Antecedente de hipertensión", ["NO", "SI"])
            diabetes = st.selectbox("Antecedente de diabetes", ["NO", "SI"])
            ant_fam_hiper = st.selectbox("Antecedentes familiares de hipertensión", ["NO", "SI"])
            tec_repro_asistida = st.selectbox("Uso de técnica de reproducción asistida", ["NO", "SI"])
            creatinina = st.number_input("Creatinina (mg/dL)", min_value=0.1, max_value=5.0, value=0.8, step=0.1)

        submitted = st.form_submit_button("Calcular riesgo")

    if submitted:
        payload = {
            "edad": edad,
            "imc": imc,
            "p_a_sistolica": p_sis,
            "p_a_diastolica": p_dia,
            "hipertension": hipertension,
            "diabetes": diabetes,
            "creatinina": creatinina,
            "ant_fam_hiper": ant_fam_hiper,
            "tec_repro_asistida": tec_repro_asistida,
        }

        pred = predict_batch(payload)[0]
        st.session_state["prediction"] = pred
        st.session_state["payload"] = payload

    if "prediction" in st.session_state:
        pred = st.session_state["prediction"]
        prob_pct = pred["proba"] * 100

        color = "#B91C1C" if pred["pred_label"] == "RIESGO" else "#15803D"
        style_box(
            f"<h3>{pred['pred_label']}</h3>"
            f"<p>Probabilidad estimada: <strong>{prob_pct:.2f}%</strong> "
            f"(umbral = {pred['threshold']:.2f})</p>",
            color,
        )

        st.markdown("#### Datos ingresados")
        st.dataframe(pd.DataFrame([st.session_state["payload"]]))

# ==========================
# TAB 2: ANÁLISIS DEL MODELO
# ==========================
with tab_model:
    st.subheader("📊 Análisis del modelo")

    # 1) Matriz de confusión desde test_metrics si existe
    cm_from_metrics = TEST_METRICS.get("confusion_matrix", None)

    if cm_from_metrics is not None:
        st.markdown("### Matriz de confusión (desde decision_policy.json)")
        cm = np.array(cm_from_metrics)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor real")
        ax.set_xticklabels(["SIN RIESGO", "RIESGO"])
        ax.set_yticklabels(["SIN RIESGO", "RIESGO"])
        st.pyplot(fig)
    else:
        st.info("No se encontró `confusion_matrix` en test_metrics.")

    # 2) Curvas ROC / PR SOLO si hay y_true + y_pred_proba
    if len(Y_TRUE) == len(Y_PROBA) and len(Y_TRUE) > 0:
        st.markdown("### Curva ROC (a partir de y_true / y_pred_proba)")
        fpr, tpr, _ = roc_curve(Y_TRUE, Y_PROBA)
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax2.plot([0, 1], [0, 1], "--", label="Azar")
        ax2.set_xlabel("FPR")
        ax2.set_ylabel("TPR")
        ax2.legend()
        st.pyplot(fig2)

        st.markdown("### Curva Precision–Recall")
        precision, recall, _ = precision_recall_curve(Y_TRUE, Y_PROBA)
        pr_auc = auc(recall, precision)
        fig3, ax3 = plt.subplots()
        ax3.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
        ax3.set_xlabel("Recall")
        ax3.set_ylabel("Precision")
        ax3.legend()
        st.pyplot(fig3)
    else:
        st.info(
            "Para curvas ROC y Precision–Recall se necesitan `y_true` y `y_pred_proba` "
            "en decision_policy.json. Ahora mismo no están definidos."
        )

# ==========================
# TAB 3: SHAP
# ==========================
with tab_shap:
    st.subheader("🧠 Interpretabilidad (SHAP)")

    if not HAS_SHAP:
        st.error("SHAP no está instalado. Añade `shap` a requirements.txt.")
    else:
        bg = get_background_for_shap()
        if bg is None:
            st.warning("No hay sample_inputs.json para usar como background.")
        else:
            st.markdown("Selecciona un paciente de ejemplo de `sample_inputs.json`")
            idx = st.number_input(
                "Índice de paciente",
                min_value=0,
                max_value=len(bg) - 1,
                value=0,
            )
            x_instance = bg.iloc[[idx]]

            st.markdown("#### Datos del paciente seleccionado")
            st.dataframe(x_instance)

            X_bg = bg.to_numpy()

            with st.spinner("Calculando valores SHAP (puede tardar unos segundos)..."):

                def predict_proba_fn(x):
                    df = pd.DataFrame(x, columns=FEATURES)
                    df = _coerce_and_align(df)
                    return PIPE.predict_proba(df)

                explainer = shap.KernelExplainer(predict_proba_fn, X_bg)
                shap_vals = explainer.shap_values(x_instance.to_numpy(), nsamples=100)

            # Clasificación binaria → lista [clase0, clase1]
            if isinstance(shap_vals, list) and len(shap_vals) >= 2:
                shap_positive = shap_vals[1]
                base_value = (
                    explainer.expected_value[1]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value
                )
            else:
                shap_positive = shap_vals
                base_value = explainer.expected_value

            st.markdown("#### Explicación local (waterfall)")

            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_positive[0],
                    base_values=base_value,
                    data=x_instance.to_numpy()[0],
                    feature_names=FEATURES,
                ),
                show=False,
            )
            st.pyplot(plt.gcf())
            plt.clf()

# ==========================
# TAB 4: DASHBOARD
# ==========================
with tab_dash:
    st.subheader("📈 Dashboard de datos (sample_inputs.json)")

    if SAMPLE_DF is None or SAMPLE_DF.empty:
        st.warning("No se encontró sample_inputs.json.")
    else:
        df = SAMPLE_DF.copy()
        df = _coerce_and_align(df)

        st.markdown("#### Vista rápida")
        st.dataframe(df.head())

        numeric_cols = [
            c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])
        ]
        if numeric_cols:
            col = st.selectbox("Variable numérica para histograma", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, bins=20, ax=ax)
            ax.set_title(f"Distribución de {col}")
            st.pyplot(fig)
        else:
            st.info("No hay columnas numéricas para graficar.")
