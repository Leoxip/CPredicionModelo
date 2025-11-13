import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.inspection import permutation_importance

# ================================
# SHAP opcional
# ================================
try:
    import shap
    HAS_SHAP = True
except:
    HAS_SHAP = False

# ================================
# Configuración general
# ================================
st.set_page_config(
    page_title="Riesgo de Preeclampsia",
    page_icon="🩺",
    layout="wide"
)
st.title("🩺 Predicción de Riesgo de Preeclampsia")
st.caption("Aplicación de apoyo. No sustituye criterio clínico.")

# ================================
# Cargar artefactos
# ================================
ART_DIR = os.path.join("artefactos", "v1")

@st.cache_resource
def load_artifacts():
    with open(os.path.join(ART_DIR, "input_schema.json")) as f:
        input_schema = json.load(f)
    with open(os.path.join(ART_DIR, "label_map.json")) as f:
        label_map = json.load(f)
    with open(os.path.join(ART_DIR, "decision_policy.json")) as f:
        policy = json.load(f)

    winner = policy["winner"]
    threshold = policy.get("threshold", 0.5)

    pipe = joblib.load(os.path.join(ART_DIR, f"pipeline_{winner}.joblib"))

    df_samples = None
    sample_path = os.path.join(ART_DIR, "sample_inputs.json")
    if os.path.exists(sample_path):
        raw = json.load(open(sample_path))
        if isinstance(raw, dict):
            raw = [raw]
        df_samples = pd.DataFrame(raw)

    rev_label = {v: k for k, v in label_map.items()}

    return (
        pipe,
        input_schema,
        label_map,
        rev_label,
        threshold,
        list(input_schema.keys()),
        policy,
        df_samples
    )

PIPE, INPUT_SCHEMA, LABEL_MAP, REV_LABEL, THRESHOLD, FEATURES, POLICY, SAMPLE_DF = load_artifacts()

TEST_METRICS = POLICY.get("test_metrics", {})
Y_TRUE = np.array(POLICY.get("y_true", []))
Y_PROBA = np.array(POLICY.get("y_pred_proba", []))


# ================================
# Funciones auxiliares
# ================================
def _coerce_and_align(df):
    for c, t in INPUT_SCHEMA.items():
        if c not in df:
            df[c] = np.nan

        if "int" in t or "float" in t:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = df[c].astype("string")

    return df[FEATURES]


def predict_batch(rec, thr=None):
    thr = THRESHOLD if thr is None else float(thr)
    df = _coerce_and_align(pd.DataFrame([rec]))
    proba = PIPE.predict_proba(df)[:, 1][0]
    pred = int(proba >= thr)

    return {
        "proba": float(proba),
        "pred_int": pred,
        "pred_label": REV_LABEL[pred],
        "threshold": thr,
    }


def style_primary_box(text, color="#0F766E"):
    st.markdown(
        f"""
        <div style="border-radius: 12px; padding: 16px;
        border: 1px solid {color}33; background-color:{color}11;">
            {text}
        </div>
        """,
        unsafe_allow_html=True
    )


def get_bg_for_shap(max_samples=100):
    if SAMPLE_DF is None: return None
    df = _coerce_and_align(SAMPLE_DF.copy())
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
    return df


# ================================
# Sidebar
# ================================
with st.sidebar:
    st.subheader("ℹ️ Modelo")
    st.write(f"Ganador: **{POLICY.get('winner')}**")
    st.write(f"Umbral: **{THRESHOLD:.2f}**")

    if TEST_METRICS:
        st.write("### Métricas")
        for k, v in TEST_METRICS.items():
            st.metric(k.upper(), f"{v:.3f}")

    st.markdown("---")
    st.write(f"SHAP: {'🟢' if HAS_SHAP else '🔴'}")


# ================================
# Tabs
# ================================
tab_pred, tab_model, tab_shap, tab_dash = st.tabs(
    ["🔮 Predicción", "📊 Análisis modelo", "🧠 Interpretabilidad", "📈 Dashboard"]
)

# ================================
# TAB 1 – Predicción
# ================================
with tab_pred:
    st.subheader("📋 Datos de la paciente")

    with st.form("form1"):
        c1, c2 = st.columns(2)

        with c1:
            edad = st.number_input("Edad", 10, 60, 30)
            imc = st.number_input("IMC", 10.0, 60.0, 25.0)
            p_sis = st.number_input("Sistólica", 70, 250, 120)
            p_dia = st.number_input("Diastólica", 40, 150, 80)

        with c2:
            hipertension = st.selectbox("Hipertensión", [0,1])
            diabetes = st.selectbox("Diabetes", [0,1])
            ant_fam_hiper = st.selectbox("Antecedente familiar HTA", [0,1])
            tec_repro = st.selectbox("Técnica reproducción asistida", [0,1])
            creatinina = st.number_input("Creatinina", 0.1, 5.0, 0.8)

        sub = st.form_submit_button("Calcular")

    if sub:
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

        pred = predict_batch(payload)
        proba = pred["proba"] * 100

        if pred["pred_label"] == "RIESGO":
            color = "#B91C1C"
        else:
            color = "#15803D"

        style_primary_box(
            f"<h3>{pred['pred_label']}</h3>"
            f"<p>Probabilidad: <b>{proba:.2f}%</b></p>",
            color
        )

        st.dataframe(pd.DataFrame([payload]))


# ================================
# TAB 2 – Análisis del modelo
# ================================
with tab_model:
    st.subheader("📊 Análisis global")

    if len(Y_TRUE) > 0:
        # Confusión
        st.markdown("### Matriz de confusión")
        cm = confusion_matrix(Y_TRUE, (Y_PROBA >= THRESHOLD).astype(int))
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        st.pyplot(fig)
        st.info(
            "Ayuda a ver cuántos casos reales de riesgo fueron correctamente detectados (TP) "
            "y cuántos se escaparon (FN)."
        )

        # ROC
        st.markdown("### Curva ROC")
        fpr, tpr, _ = roc_curve(Y_TRUE, Y_PROBA)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr)
        ax.plot([0,1],[0,1],"--")
        st.pyplot(fig)
        st.info("Mientras más se acerque a la esquina superior izquierda, mejor el modelo.")

        # PR
        st.markdown("### Curva Precision–Recall")
        precision, recall, _ = precision_recall_curve(Y_TRUE, Y_PROBA)
        fig, ax = plt.subplots()
        ax.plot(recall, precision)
        st.pyplot(fig)
        st.info("Muestra el equilibrio entre detectar casos (recall) y evitar falsos positivos (precision).")

        # IMPORTANCIA GLOBAL (Permutation Importance)
        st.markdown("### Importancia global de variables (Permutation Importance)")
        X = _coerce_and_align(SAMPLE_DF.copy())
        r = permutation_importance(PIPE, X, Y_PROBA >= THRESHOLD, n_repeats=10, random_state=42)

        imp_df = pd.DataFrame({
            "feature": FEATURES,
            "importance": r.importances_mean
        }).sort_values("importance", ascending=False)

        fig, ax = plt.subplots(figsize=(6,4))
        sns.barplot(data=imp_df, x="importance", y="feature", ax=ax)
        st.pyplot(fig)
        st.info(
            "Mide cuánto empeora el modelo cuando se mezcla cada variable. "
            "Cuanto más alta → más importante para la predicción global."
        )


# ================================
# TAB 3 – Interpretabilidad
# ================================
with tab_shap:
    st.subheader("🧠 Interpretabilidad con SHAP")
    if not HAS_SHAP:
        st.warning("Instala SHAP.")
    else:
        bg = get_bg_for_shap()
        if bg is None:
            st.warning("No hay sample_inputs.json")
        else:
            idx = st.number_input("Paciente", 0, len(bg)-1, 0)
            x = bg.iloc[[idx]]

            with st.spinner("Calculando SHAP..."):

                def pred_fn(xarr):
                    df = pd.DataFrame(xarr, columns=FEATURES)
                    return PIPE.predict_proba(_coerce_and_align(df))

                explainer = shap.KernelExplainer(pred_fn, bg.to_numpy())
                sv = explainer.shap_values(x.to_numpy(), nsamples=100)

            sv_pos = sv[1] if isinstance(sv, list) else sv

            shap.waterfall_plot(
                shap.Explanation(
                    values=sv_pos[0],
                    base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value,
                    data=x.to_numpy()[0],
                    feature_names=FEATURES
                )
            )
            st.pyplot(plt.gcf())
            plt.clf()


# ================================
# TAB 4 – Dashboard
# ================================
with tab_dash:
    st.subheader("📈 Dashboard exploratorio")

    if SAMPLE_DF is None:
        st.warning("No hay sample_inputs.json")
    else:
        df = _coerce_and_align(SAMPLE_DF.copy())

        st.markdown("### Distribución por variable")
        col = st.selectbox("Variable numérica", df.columns)

        fig, ax = plt.subplots()
        sns.histplot(df[col], kde=True, ax=ax)
        st.pyplot(fig)
        st.info("Permite ver la forma de la distribución clínica.")

        st.markdown("### Correlación entre variables")
        fig, ax = plt.subplots(figsize=(7,5))
        sns.heatmap(df.corr(), cmap="coolwarm", annot=False, ax=ax)
        st.pyplot(fig)
        st.info("Ayuda a detectar variables clínicas fuertemente asociadas entre sí.")

        st.markdown("### Relación entre dos variables")
        c1, c2 = st.columns(2)
        xvar = c1.selectbox("Variable X", df.columns)
        yvar = c2.selectbox("Variable Y", df.columns)

        fig, ax = plt.subplots()
        sns.regplot(data=df, x=xvar, y=yvar, ax=ax)
        st.pyplot(fig)
        st.info("Permite ver tendencias clínicas entre dos mediciones.")
