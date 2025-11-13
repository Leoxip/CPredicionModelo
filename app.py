import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

# ============================================================
# CONFIGURACIÓN GENERAL
# ============================================================
st.set_page_config(
    page_title="Riesgo de Preeclampsia",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Predicción de Riesgo de Preeclampsia")
st.caption("Esta herramienta de IA **no reemplaza la evaluación médica profesional**.")


# ============================================================
# CARGAR ARTEFACTOS
# ============================================================
ART_DIR = os.path.join("artefactos", "v1")

@st.cache_resource
def load_artifacts():
    with open(os.path.join(ART_DIR, "input_schema.json"), "r", encoding="utf-8") as f:
        input_schema = json.load(f)

    with open(os.path.join(ART_DIR, "label_map.json"), "r", encoding="utf-8") as f:
        label_map = json.load(f)

    with open(os.path.join(ART_DIR, "decision_policy.json"), "r", encoding="utf-8") as f:
        policy = json.load(f)

    winner = policy["winner"]
    threshold = float(policy.get("threshold", 0.5))

    pipe = joblib.load(os.path.join(ART_DIR, f"pipeline_{winner}.joblib"))

    sample_df = None
    samples_path = os.path.join(ART_DIR, "sample_inputs.json")
    if os.path.exists(samples_path):
        raw = json.load(open(samples_path))
        if isinstance(raw, dict):
            raw = [raw]
        sample_df = pd.DataFrame(raw)

    rev_label = {v: k for k, v in label_map.items()}

    return pipe, input_schema, label_map, rev_label, threshold, policy, sample_df


PIPE, INPUT_SCHEMA, LABEL_MAP, REV_LABEL, THRESHOLD, POLICY, SAMPLE_DF = load_artifacts()

Y_TRUE = np.array(POLICY.get("y_true", []))
Y_PROBA = np.array(POLICY.get("y_pred_proba", []))
TEST_METRICS = POLICY.get("test_metrics", {})

FEATURES = list(INPUT_SCHEMA.keys())

# ============================================================
# FUNCIONES AUXILIARES
# ============================================================
def _coerce_and_align(df):
    """Asegurar tipos y orden correcto."""
    for col, t in INPUT_SCHEMA.items():
        if col not in df:
            df[col] = np.nan

        t = str(t).lower()
        if t.startswith(("int", "float")):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif t in ("bool", "boolean"):
            df[col] = df[col].astype(bool)
        else:
            df[col] = df[col].astype("string")

    return df[FEATURES]


def predict_batch(payload, thr=None):
    thr = THRESHOLD if thr is None else float(thr)

    df = _coerce_and_align(pd.DataFrame([payload]))
    proba = PIPE.predict_proba(df)[:, 1][0]
    pred = int(proba >= thr)

    return {
        "proba": float(proba),
        "pred_int": pred,
        "pred_label": REV_LABEL[pred],
        "threshold": thr,
    }


def get_background_for_shap(max_samples=120):
    if SAMPLE_DF is None:
        return None
    df = _coerce_and_align(SAMPLE_DF.copy())
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
    return df


# ============================================================
# SIDE BAR
# ============================================================
with st.sidebar:
    st.header("ℹ️ Información del modelo")
    st.write(f"**Modelo ganador:** `{POLICY.get('winner', 'N/A')}`")
    st.write(f"**Umbral:** `{THRESHOLD:.2f}`")

    if TEST_METRICS:
        st.subheader("Métricas en test")
        for k, v in TEST_METRICS.items():
            st.metric(k.upper(), f"{v:.3f}")

# ============================================================
# TABS
# ============================================================
tab_pred, tab_model, tab_shap, tab_dash = st.tabs(
    ["🔮 Predicción", "📊 Análisis del modelo", "🧠 Interpretabilidad (SHAP)", "📈 Dashboard"]
)

# ============================================================
# TAB 1 - PREDICCIÓN
# ============================================================
with tab_pred:
    st.subheader("📋 Datos clínicos")

    with st.form("form_paciente"):
        col1, col2 = st.columns(2)

        with col1:
            edad = st.number_input("Edad", 10, 60, 25)
            imc = st.number_input("IMC", 10.0, 60.0, 25.0, 0.1)
            p_sis = st.number_input("Presión sistólica", 70, 250, 120)
            p_dia = st.number_input("Presión diastólica", 40, 150, 80)

        with col2:
            hipertension = st.selectbox("Hipertensión previa", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")
            diabetes = st.selectbox("Diabetes", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")
            ant_fam_hiper = st.selectbox("Antecedente familiar de HTA", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")
            tec_repro = st.selectbox("Técnica de reproducción asistida", [0, 1], format_func=lambda x: "No" if x==0 else "Sí")
            creatinina = st.number_input("Creatinina", 0.1, 5.0, 0.8, 0.1)

        submit = st.form_submit_button("Calcular riesgo")

    if submit:
        payload = {
            "edad": edad,
            "imc": imc,
            "p_a_sistolica": p_sis,
            "p_a_diastolica": p_dia,
            "hipertension": hipertension,
            "diabetes": diabetes,
            "creatinina": creatinina,
            "ant_fam_hiper": ant_fam_hiper,
            "tec_repro_asistida": tec_repro,
        }

        result = predict_batch(payload)
        st.session_state["payload"] = payload
        st.session_state["prediction"] = result

    if "prediction" in st.session_state:
        pred = st.session_state["prediction"]
        payload = st.session_state["payload"]

        st.markdown("---")
        st.subheader("🔍 Resultado del modelo")

        color = "#B91C1C" if pred["pred_label"] == "RIESGO" else "#15803D"

        st.markdown(
            f"""
            <div style="padding:15px;border-radius:10px;background:{color}22;border:1px solid {color};">
                <h3 style="margin:0;color:{color};">{pred['pred_label']}</h3>
                <p style="margin:0;">Probabilidad: <b>{pred['proba']*100:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.write("### Datos ingresados")
        st.dataframe(pd.DataFrame([payload]))

# ============================================================
# TAB 2 - ANÁLISIS DEL MODELO
# ============================================================
with tab_model:
    st.subheader("📊 Evaluación del modelo")

    if len(Y_TRUE) == len(Y_PROBA) and len(Y_TRUE) > 0:

        y_pred = (Y_PROBA >= THRESHOLD).astype(int)

        # MATRIZ CONFUSIÓN
        st.write("### Matriz de confusión")
        cm = confusion_matrix(Y_TRUE, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d", ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        st.pyplot(fig)

        # ROC
        st.write("### Curva ROC")
        fpr, tpr, _ = roc_curve(Y_TRUE, Y_PROBA)
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax2.plot([0,1],[0,1],"--")
        st.pyplot(fig2)

        # Precision Recall
        st.write("### Curva Precision–Recall")
        prec, rec, _ = precision_recall_curve(Y_TRUE, Y_PROBA)
        pr_auc = auc(rec, prec)
        fig3, ax3 = plt.subplots()
        ax3.plot(rec, prec, label=f"AUC = {pr_auc:.3f}")
        st.pyplot(fig3)

# ============================================================
# TAB 3 - SHAP
# ============================================================
with tab_shap:
    st.subheader("🧠 Interpretabilidad SHAP")

    bg = get_background_for_shap()
    if bg is None:
        st.warning("No hay sample_inputs.json para calcular SHAP.")
    else:
        X_bg = bg.to_numpy()

        idx = st.number_input("Paciente índice", 0, len(bg)-1, 0)
        x_instance = bg.iloc[[idx]]

        st.write("### Valores de entrada")
        st.dataframe(x_instance)

        st.write("### Cálculo SHAP (KernelExplainer)")
        with st.spinner("Calculando..."):

            def predict_fn(x):
                df = _coerce_and_align(pd.DataFrame(x, columns=FEATURES))
                return PIPE.predict_proba(df)

            explainer = shap.KernelExplainer(predict_fn, X_bg)
            shap_vals = explainer.shap_values(x_instance.to_numpy(), nsamples=100)

        # Seleccionar clase positiva
        if isinstance(shap_vals, list) and len(shap_vals)==2:
            shap_pos = shap_vals[1]
        else:
            shap_pos = shap_vals

        # Base value fijo
        if isinstance(explainer.expected_value, (list, np.ndarray)):
            base_val = explainer.expected_value[1]
        else:
            base_val = explainer.expected_value

        exp = shap.Explanation(
            values=shap_pos[0],
            base_values=float(base_val),
            data=x_instance.to_numpy()[0],
            feature_names=FEATURES
        )

        shap.plots.waterfall(exp, show=False)
        st.pyplot(plt.gcf())
        plt.clf()

# ============================================================
# TAB 4 - DASHBOARD
# ============================================================
with tab_dash:
    if SAMPLE_DF is None:
        st.warning("No hay sample_inputs.json")
    else:
        st.subheader("📈 Exploración de datos")

        df = _coerce_and_align(SAMPLE_DF.copy())

        st.write("### Vista previa")
        st.dataframe(df.head())

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        if numeric_cols:
            col = st.selectbox("Variable numérica", numeric_cols)

            st.write("### Histograma")
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, ax=ax)
            st.pyplot(fig)

            st.write("### Boxplot")
            fig2, ax2 = plt.subplots()
            sns.boxplot(x=df[col], ax=ax2)
            st.pyplot(fig2)

            st.write("### Distribución vs otra variable")
            other = st.selectbox("Comparar contra:", numeric_cols, index=1)
            fig3, ax3 = plt.subplots()
            sns.scatterplot(x=df[col], y=df[other], ax=ax3)
            st.pyplot(fig3)
