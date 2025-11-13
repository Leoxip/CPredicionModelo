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

# =============== CONFIG STREAMLIT ===============
st.set_page_config(
    page_title="Riesgo de Preeclampsia",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Predicción de Riesgo de Preeclampsia")
st.caption("Aplicación de apoyo clínico — no reemplaza evaluación médica.")

# =============== INTENTAR IMPORTAR SHAP ===============
try:
    import shap
    HAS_SHAP = True
except:
    HAS_SHAP = False

# =============== CARGA DE ARTEFACTOS ===============
ART_DIR = os.path.join("artefactos", "v1")

@st.cache_resource
def load_artifacts():

    # Rutas
    input_schema = json.load(open(os.path.join(ART_DIR, "input_schema.json"), "r"))
    label_map = json.load(open(os.path.join(ART_DIR, "label_map.json"), "r"))
    policy = json.load(open(os.path.join(ART_DIR, "decision_policy.json"), "r"))

    # Modelo ganador
    winner = policy["winner"]
    threshold = float(policy.get("threshold", 0.5))

    pipe_path = os.path.join(ART_DIR, f"pipeline_{winner}.joblib")
    pipe = joblib.load(pipe_path)

    # Muestras para dashboard / SHAP
    sample_df = None
    sample_path = os.path.join(ART_DIR, "sample_inputs.json")
    if os.path.exists(sample_path):
        raw = json.load(open(sample_path, "r"))
        if isinstance(raw, dict):
            raw = [raw]
        sample_df = pd.DataFrame(raw)

    rev_label = {v: k for k, v in label_map.items()}
    features = list(input_schema.keys())

    return pipe, input_schema, label_map, rev_label, threshold, features, policy, sample_df


PIPE, INPUT_SCHEMA, LABEL_MAP, REV_LABEL, THRESHOLD, FEATURES, POLICY, SAMPLE_DF = load_artifacts()

TEST_METRICS = POLICY.get("test_metrics", {})
Y_TRUE = np.array(POLICY.get("y_true", [])) if "y_true" in POLICY else None
Y_PROBA = np.array(POLICY.get("y_pred_proba", [])) if "y_pred_proba" in POLICY else None


# =============== FUNCIONES ===============
def _coerce_and_align(df):
    for col, t in INPUT_SCHEMA.items():
        if col not in df:
            df[col] = np.nan

        ts = str(t).lower()
        if ts.startswith("int") or ts.startswith("float"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        elif ts in ("bool", "boolean"):
            df[col] = df[col].astype("bool")
        else:
            df[col] = df[col].astype("string")

    return df[FEATURES]


def predict_batch(records, thr=None):
    thr = THRESHOLD if thr is None else float(thr)

    if isinstance(records, dict):
        records = [records]

    df = _coerce_and_align(pd.DataFrame(records))

    proba = PIPE.predict_proba(df)[:, 1]
    preds = (proba >= thr).astype(int)

    results = []
    for p, y in zip(proba, preds):
        results.append({
            "proba": float(p),
            "pred_int": int(y),
            "pred_label": REV_LABEL[int(y)],
            "threshold": thr,
        })
    return results


def get_background_for_shap(max_samples=100):
    if SAMPLE_DF is None or SAMPLE_DF.empty:
        return None
    df = _coerce_and_align(SAMPLE_DF.copy())
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
    return df


def style_box(text, color):
    st.markdown(
        f"""
        <div style="padding:15px;border-radius:10px;background:{color}20;
                    border:1px solid {color}55;margin-bottom:15px;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


# =============== SIDEBAR ===============
with st.sidebar:
    st.header("ℹ️ Modelo")
    st.markdown(f"**Modelo ganador:** `{POLICY.get('winner')}`")
    st.markdown(f"**Umbral:** `{THRESHOLD:.2f}`")

    st.markdown("### Métricas en test")
    if TEST_METRICS:
        for k, v in TEST_METRICS.items():
            try:
                val = float(v)
                st.metric(k.upper(), f"{val:.3f}")
            except:
                st.metric(k.upper(), "N/A")
    else:
        st.info("No hay métricas disponibles.")

    st.markdown("---")
    st.markdown(f"SHAP disponible: {'✅' if HAS_SHAP else '❌'}")


# =============== PESTAÑAS ===============
tab_pred, tab_model, tab_shap, tab_dash = st.tabs(
    ["🔮 Predicción", "📊 Análisis del modelo", "🧠 Interpretabilidad (SHAP)", "📈 Dashboard"]
)

# =============== TAB 1: PREDICCIÓN ===============
with tab_pred:
    st.subheader("📋 Datos de la paciente")

    with st.form("form"):
        col1, col2 = st.columns(2)

        with col1:
            edad = st.number_input("Edad", 10, 60, 30)
            imc = st.number_input("IMC", 10.0, 60.0, 25.0)
            p_sis = st.number_input("Presión sistólica", 70, 250, 120)
            p_dia = st.number_input("Presión diastólica", 40, 150, 80)

        with col2:
            hipertension = st.selectbox("Hipertensión previa", [0, 1])
            diabetes = st.selectbox("Diabetes previa", [0, 1])
            ant_fam = st.selectbox("Ant. familiares hipertensión", [0, 1])
            repro_asist = st.selectbox("Reproducción asistida", [0, 1])
            creat = st.number_input("Creatinina", 0.1, 5.0, 0.8)

        submit = st.form_submit_button("Calcular riesgo")

    if submit:
        payload = {
            "edad": edad,
            "imc": imc,
            "p_a_sistolica": p_sis,
            "p_a_diastolica": p_dia,
            "hipertension": hipertension,
            "diabetes": diabetes,
            "creatinina": creat,
            "ant_fam_hiper": ant_fam,
            "tec_repro_asistida": repro_asist,
        }

        pred = predict_batch(payload)[0]

        st.session_state["prediction"] = pred
        st.session_state["payload"] = payload

    if "prediction" in st.session_state:
        pred = st.session_state["prediction"]
        prob = pred["proba"] * 100

        color = "#B91C1C" if pred["pred_label"] == "RIESGO" else "#15803D"
        style_box(
            f"<h3>{pred['pred_label']}</h3>"
            f"<p>Probabilidad estimada: <strong>{prob:.2f}%</strong></p>",
            color
        )

        st.dataframe(pd.DataFrame([st.session_state["payload"]]))

# =============== TAB 2: ANÁLISIS ===============
with tab_model:
    st.subheader("📊 Análisis del modelo")

    if (
        Y_TRUE is not None
        and Y_PROBA is not None
        and len(Y_TRUE) == len(Y_PROBA)
        and len(Y_TRUE) > 0
    ):
        # MATRIZ DE CONFUSIÓN
        st.markdown("### Matriz de confusión")
        y_pred = (Y_PROBA >= THRESHOLD).astype(int)
        cm = confusion_matrix(Y_TRUE, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
        st.pyplot(fig)

        # ROC
        st.markdown("### Curva ROC")
        fpr, tpr, _ = roc_curve(Y_TRUE, Y_PROBA)
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax2.plot([0,1], [0,1], "--")
        st.pyplot(fig2)

        # PR Curve
        st.markdown("### Curva Precision–Recall")
        prec, rec, _ = precision_recall_curve(Y_TRUE, Y_PROBA)
        pr_auc = auc(rec, prec)
        fig3, ax3 = plt.subplots()
        ax3.plot(rec, prec, label=f"AUC = {pr_auc:.3f}")
        st.pyplot(fig3)

    else:
        st.warning("No hay datos de test en decision_policy.json")

# =============== TAB 3: SHAP ===============
with tab_shap:
    st.subheader("🧠 Interpretabilidad con SHAP")

    if not HAS_SHAP:
        st.error("Instala `shap` para usar esta sección.")
    else:
        bg = get_background_for_shap()

        if bg is None:
            st.warning("No existe sample_inputs.json para usar SHAP.")
        else:
            idx = st.number_input("Índice de paciente", 0, len(bg)-1, 0)
            x_inst = bg.iloc[[idx]]

            st.write("Paciente seleccionado:")
            st.dataframe(x_inst)

            X_bg = bg.to_numpy()

            with st.spinner("Calculando SHAP..."):
                def pred_fn(x):
                    df = pd.DataFrame(x, columns=FEATURES)
                    df = _coerce_and_align(df)
                    return PIPE.predict_proba(df)

                explainer = shap.KernelExplainer(pred_fn, X_bg)
                shap_vals = explainer.shap_values(x_inst.to_numpy(), nsamples=100)

            # clase positiva
            if isinstance(shap_vals, list):
                shap_pos = shap_vals[1]
                base = explainer.expected_value[1]
            else:
                shap_pos = shap_vals
                base = explainer.expected_value

            st.markdown("### Explicación local")

            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_pos[0],
                    base_values=base,
                    data=x_inst.to_numpy()[0],
                    feature_names=FEATURES,
                ),
                show=False
            )

            st.pyplot(plt.gcf())
            plt.clf()

# =============== TAB 4: DASHBOARD ===============
with tab_dash:
    st.subheader("📈 Dashboard de datos")

    if SAMPLE_DF is None or SAMPLE_DF.empty:
        st.warning("No hay sample_inputs.json para dashboard.")
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
            st.info("No hay columnas numéricas para graficar.")
