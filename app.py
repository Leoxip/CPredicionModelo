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
# IMPORTAR SHAP
# ==========================
try:
    import shap
    HAS_SHAP = True
except:
    HAS_SHAP = False

# ==========================
# CARGA DE ARTEFACTOS
# ==========================
ART_DIR = os.path.join("artefactos", "v1")

@st.cache_resource
def load_artifacts():

    input_schema = json.load(open(os.path.join(ART_DIR, "input_schema.json")))
    label_map = json.load(open(os.path.join(ART_DIR, "label_map.json")))
    policy = json.load(open(os.path.join(ART_DIR, "decision_policy.json")))

    winner = policy["winner"]
    threshold = float(policy.get("threshold", 0.5))

    pipe_path = os.path.join(ART_DIR, f"pipeline_{winner}.joblib")
    pipe = joblib.load(pipe_path)

    sample_df = None
    sample_path = os.path.join(ART_DIR, "sample_inputs.json")
    if os.path.exists(sample_path):
        raw = json.load(open(sample_path))
        if isinstance(raw, dict):
            raw = [raw]
        sample_df = pd.DataFrame(raw)

    rev_label = {v: k for k, v in label_map.items()}
    features = list(input_schema.keys())

    return pipe, input_schema, label_map, rev_label, threshold, features, policy, sample_df

PIPE, INPUT_SCHEMA, LABEL_MAP, REV_LABEL, THRESHOLD, FEATURES, POLICY, SAMPLE_DF = load_artifacts()

TEST_METRICS = POLICY.get("test_metrics", {})
Y_TRUE = np.array(POLICY.get("y_true", []))
Y_PROBA = np.array(POLICY.get("y_pred_proba", []))

# ==========================
# FUNCIONES AUXILIARES
# ==========================
def normalize_yes_no(x):
    """Convierte 'SI'/'NO' a 1/0."""
    if str(x).strip().upper() == "SI":
        return 1
    if str(x).strip().upper() == "NO":
        return 0
    return x

def _coerce_and_align(df):
    for col, typ in INPUT_SCHEMA.items():

        if col not in df.columns:
            df[col] = np.nan

        if typ in ("int64", "float64"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = df[col].apply(normalize_yes_no).astype("float")
    return df[FEATURES]

def predict_batch(records, thr=None):
    thr = THRESHOLD if thr is None else float(thr)

    if isinstance(records, dict):
        records = [records]

    df = pd.DataFrame(records)
    df = _coerce_and_align(df)

    proba = PIPE.predict_proba(df)[:, 1]
    preds = (proba >= thr).astype(int)

    return [
        {
            "proba": float(p),
            "pred_int": int(c),
            "pred_label": REV_LABEL[int(c)],
            "threshold": thr
        }
        for p, c in zip(proba, preds)
    ]

def get_background_for_shap(max_samples=100):
    if SAMPLE_DF is None or SAMPLE_DF.empty:
        return None

    df = SAMPLE_DF.copy()
    df = _coerce_and_align(df)

    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)

    return df

# ==========================
# SIDEBAR
# ==========================
with st.sidebar:
    st.header("ℹ️ Modelo")
    st.markdown(f"**Modelo ganador:** `{POLICY.get('winner')}`")
    st.markdown(f"**Umbral:** `{THRESHOLD:.2f}`")

    st.markdown("### Métricas del test")
    if TEST_METRICS:
        for k, v in TEST_METRICS.items():
            if isinstance(v, (int, float)):
                st.metric(k.upper(), f"{v:.3f}")
    else:
        st.info("No hay métricas disponibles.")

    st.markdown("---")
    st.markdown(f"SHAP disponible: {'🟢' if HAS_SHAP else '🔴'}")

# ==========================
# TABS
# ==========================
tab_pred, tab_model, tab_shap, tab_dash = st.tabs(
    ["🔮 Predicción", "📊 Análisis del modelo", "🧠 SHAP", "📈 Dashboard"]
)

# ==========================
# TAB 1: PREDICCIÓN
# ==========================
with tab_pred:
    st.subheader("📋 Datos de la paciente")

    with st.form("form_input"):
        c1, c2 = st.columns(2)

        with c1:
            edad = st.number_input("Edad", 10, 60, 30)
            imc = st.number_input("IMC", 10.0, 60.0, 25.0)
            sist = st.number_input("Presión sistólica", 70, 250, 120)
            diast = st.number_input("Presión diastólica", 40, 150, 80)

        with c2:
            hipert = st.selectbox("Hipertensión", ["NO", "SI"])
            diab = st.selectbox("Diabetes", ["NO", "SI"])
            antfam = st.selectbox("Ant. familiares HTA", ["NO", "SI"])
            repro = st.selectbox("Reproducción asistida", ["NO", "SI"])
            creat = st.number_input("Creatinina", 0.1, 5.0, 0.8)

        enviar = st.form_submit_button("Calcular riesgo")

    if enviar:
        payload = {
            "edad": edad,
            "imc": imc,
            "p_a_sistolica": sist,
            "p_a_diastolica": diast,
            "hipertension": hipert,
            "diabetes": diab,
            "creatinina": creat,
            "ant_fam_hiper": antfam,
            "tec_repro_asistida": repro
        }

        pred = predict_batch(payload)[0]
        st.session_state["pred"] = pred
        st.session_state["payload"] = payload

    if "pred" in st.session_state:
        p = st.session_state["pred"]
        prob = p["proba"] * 100
        color = "#B91C1C" if p["pred_label"] == "RIESGO" else "#15803D"

        st.markdown(
            f"""
            <div style='padding:14px;background:{color}22;border-radius:12px;'>
            <h3>{p['pred_label']}</h3>
            <p>Probabilidad estimada: <b>{prob:.2f}%</b></p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.dataframe(pd.DataFrame([st.session_state["payload"]]))

# ==========================
# TAB 2: ANÁLISIS
# ==========================
with tab_model:
    st.subheader("📊 Análisis del modelo")

    if len(Y_TRUE) == 0:
        st.warning("No hay datos de test en decision_policy.json")
    else:
        y_pred = (Y_PROBA >= THRESHOLD).astype(int)

        st.markdown("### Matriz de confusión")
        cm = confusion_matrix(Y_TRUE, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)

        st.markdown("### ROC")
        fpr, tpr, _ = roc_curve(Y_TRUE, Y_PROBA)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr)
        ax2.set_title(f"AUC ROC = {auc(fpr,tpr):.3f}")
        st.pyplot(fig2)

        st.markdown("### Precision–Recall")
        prec, rec, _ = precision_recall_curve(Y_TRUE, Y_PROBA)
        fig3, ax3 = plt.subplots()
        ax3.plot(rec, prec)
        ax3.set_title(f"AUC PR = {auc(rec,prec):.3f}")
        st.pyplot(fig3)

# ==========================
# TAB 3: SHAP
# ==========================
with tab_shap:

    st.subheader("🧠 Interpretabilidad con SHAP")

    if not HAS_SHAP:
        st.error("Necesitas instalar shap en requirements.txt")
    else:
        bg = get_background_for_shap()

        if bg is None:
            st.warning("sample_inputs.json no existe.")
        else:
            idx = st.number_input("Índice", 0, len(bg)-1, 0)
            x_inst = bg.iloc[[idx]]

            X_bg = bg.to_numpy()

            with st.spinner("Calculando SHAP..."):
                def pred_fn(x):
                    df = pd.DataFrame(x, columns=FEATURES)
                    df = _coerce_and_align(df)
                    return PIPE.predict_proba(df)

                explainer = shap.KernelExplainer(pred_fn, X_bg)
                shap_vals = explainer.shap_values(x_inst.to_numpy(), nsamples=100)

            if isinstance(shap_vals, list):
                shap_pos = shap_vals[1]
                base = explainer.expected_value[1]
            else:
                shap_pos = shap_vals
                base = explainer.expected_value

            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_pos[0],
                    base_values=base,
                    data=x_inst.to_numpy()[0],
                    feature_names=FEATURES
                ),
                show=False
            )
            st.pyplot(plt.gcf())
            plt.clf()

# ==========================
# TAB 4: DASHBOARD
# ==========================
with tab_dash:

    st.subheader("📈 Dashboard exploratorio")

    if SAMPLE_DF is None:
        st.warning("No hay sample_inputs.json")
    else:
        df = SAMPLE_DF.copy()
        df = _coerce_and_align(df)

        st.dataframe(df.head())

        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        if numeric_cols:
            col = st.selectbox("Variable numérica", numeric_cols)
            fig, ax = plt.subplots()
            sns.histplot(df[col], kde=True, bins=20, ax=ax)
            st.pyplot(fig)
        else:
            st.info("No hay columnas numéricas.")
