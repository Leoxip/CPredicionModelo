import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)

# ================================
# Intentamos importar SHAP
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
    layout="wide",
)

st.title("🩺 Predicción de Riesgo de Preeclampsia")
st.caption("Aplicación de apoyo clínico — No sustituye la evaluación médica.")


# ================================
# Cargar artefactos
# ================================
ART_DIR = os.path.join("artefactos", "v1")


@st.cache_resource
def load_artifacts():

    input_schema_path = os.path.join(ART_DIR, "input_schema.json")
    label_map_path = os.path.join(ART_DIR, "label_map.json")
    policy_path = os.path.join(ART_DIR, "decision_policy.json")
    sample_inputs_path = os.path.join(ART_DIR, "sample_inputs.json")

    with open(input_schema_path, "r", encoding="utf-8") as f:
        input_schema = json.load(f)
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    with open(policy_path, "r", encoding="utf-8") as f:
        policy = json.load(f)

    winner_name = policy["winner"]
    threshold = float(policy.get("threshold", 0.5))

    pipe = joblib.load(os.path.join(ART_DIR, f"pipeline_{winner_name}.joblib"))

    sample_df = None
    if os.path.exists(sample_inputs_path):
        with open(sample_inputs_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        sample_df = pd.DataFrame(raw if isinstance(raw, list) else [raw])

    rev_label = {v: k for k, v in label_map.items()}
    features = list(input_schema.keys())

    return pipe, input_schema, label_map, rev_label, threshold, features, policy, sample_df


PIPE, INPUT_SCHEMA, LABEL_MAP, REV_LABEL, THRESHOLD, FEATURES, POLICY, SAMPLE_DF = load_artifacts()

TEST_METRICS = POLICY.get("test_metrics", {})
Y_TRUE = np.array(POLICY.get("y_true", [])) if "y_true" in POLICY else None
Y_PROBA = np.array(POLICY.get("y_pred_proba", [])) if "y_pred_proba" in POLICY else None


# ================================
# Funciones auxiliares
# ================================
def _coerce_and_align(df):
    for c, t in INPUT_SCHEMA.items():
        if c not in df.columns:
            df[c] = np.nan

        if str(t).lower().startswith(("int", "float")):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif str(t).lower() in ("bool", "boolean"):
            df[c] = df[c].astype(bool)
        else:
            df[c] = df[c].astype("string")

    return df[FEATURES]


def predict_batch(records, thr=None):
    thr = THRESHOLD if thr is None else float(thr)

    if isinstance(records, dict):
        records = [records]

    df = _coerce_and_align(pd.DataFrame(records))
    proba = PIPE.predict_proba(df)[:, 1]
    preds = (proba >= thr).astype(int)

    return [{
        "proba": float(p),
        "pred_int": int(y),
        "pred_label": REV_LABEL[int(y)],
        "threshold": thr,
    } for p, y in zip(proba, preds)]


# =====================================
# Estilos visuales
# =====================================
def style_box(title, value, color="#0369A1"):
    st.markdown(
        f"""
        <div style="
            background: #F0F9FF;
            border-left: 6px solid {color};
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 10px;">
            <h4 style="margin:0; color:{color};">{title}</h4>
            <p style="margin:0; font-size:22px; font-weight:bold;">{value}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def get_background_for_shap(max_samples=100):
    if SAMPLE_DF is None:
        return None

    df = _coerce_and_align(SAMPLE_DF.copy())
    return df.sample(min(len(df), max_samples), random_state=42)


# ================================
# SIDEBAR
# ================================
with st.sidebar:
    st.header("ℹ️ Datos del modelo")
    st.write(f"**Modelo:** {POLICY.get('winner')}")
    st.write(f"**Umbral:** {THRESHOLD:.2f}")

    st.markdown("---")
    st.write("SHAP disponible:", "✅" if HAS_SHAP else "❌")


# ================================
# TABS
# ================================
tab_pred, tab_model, tab_shap, tab_dash = st.tabs([
    "🔮 Predicción",
    "📊 Análisis del modelo",
    "🧠 Interpretabilidad",
    "📈 Dashboard",
])


# ================================
# TAB 1 – PREDICCIÓN
# ================================
with tab_pred:
    st.subheader("📋 Datos clínicos de la paciente")

    with st.form("form_paciente"):
        col1, col2 = st.columns(2)

        with col1:
            edad = st.number_input("Edad", 10, 60, 30)
            imc = st.number_input("IMC", 10.0, 60.0, 25.0)
            p_sis = st.number_input("Presión sistólica", 70, 250, 120)
            p_dia = st.number_input("Presión diastólica", 40, 150, 80)

        with col2:
            hipert = st.selectbox("Hipertensión previa", [0, 1])
            diab = st.selectbox("Diabetes", [0, 1])
            ant_fam = st.selectbox("Antecedente familiar HTA", [0, 1])
            repro = st.selectbox("Reproducción asistida", [0, 1])
            crea = st.number_input("Creatinina", 0.1, 5.0, 0.8)

        submitted = st.form_submit_button("Calcular riesgo")

    if submitted:
        payload = {
            "edad": edad,
            "imc": imc,
            "p_a_sistolica": p_sis,
            "p_a_diastolica": p_dia,
            "hipertension": hipert,
            "diabetes": diab,
            "creatinina": crea,
            "ant_fam_hiper": ant_fam,
            "tec_repro_asistida": repro,
        }

        result = predict_batch(payload)[0]
        st.session_state["prediction_dict"] = result
        st.session_state["payload"] = payload

    pred = st.session_state.get("prediction_dict")

    if pred:
        st.subheader("🔍 Resultado")

        style_box("Probabilidad de riesgo", f"{pred['proba']*100:.2f}%")
        style_box("Clasificación", pred["pred_label"])
        style_box("Umbral", pred["threshold"])

        st.markdown("### Datos ingresados")
        st.dataframe(pd.DataFrame([st.session_state["payload"]]))


# ================================
# TAB 2 – ANÁLISIS GLOBAL DEL MODELO
# ================================
with tab_model:
    st.subheader("📊 Evaluación del modelo")

    if not TEST_METRICS:
        st.warning("No hay métricas en decision_policy.json")
    else:
        st.markdown("### 📌 Métricas")
        cols = st.columns(3)
        i = 0
        for k, v in TEST_METRICS.items():
            with cols[i % 3]:
                try:
                    val = f"{float(v):.3f}"
                except:
                    val = str(v)

                style_box(k.upper(), val)
            i += 1

    if (
        Y_TRUE is not None and Y_PROBA is not None and
        len(Y_TRUE) == len(Y_PROBA) and len(Y_TRUE) > 0
    ):
        st.markdown("### 🧩 Matriz de confusión")

        cm = confusion_matrix(Y_TRUE, (Y_PROBA >= THRESHOLD).astype(int))
        fig, ax = plt.subplots()
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["No Riesgo", "Riesgo"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["No Riesgo", "Riesgo"])
        st.pyplot(fig)

        st.markdown("### 📈 Curva ROC")
        fpr, tpr, _ = roc_curve(Y_TRUE, Y_PROBA)
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax2.plot([0, 1], [0, 1], "--")
        st.pyplot(fig2)

        st.markdown("### 📈 Curva Precision–Recall")
        prec, rec, _ = precision_recall_curve(Y_TRUE, Y_PROBA)
        pr_auc = auc(rec, prec)
        fig3, ax3 = plt.subplots()
        ax3.plot(rec, prec, label=f"AUC = {pr_auc:.3f}")
        st.pyplot(fig3)


# ================================
# TAB 3 – SHAP
# ================================
with tab_shap:
    st.subheader("🧠 Interpretabilidad local")

    if not HAS_SHAP:
        st.warning("Instala `shap` para usar esta sección.")
    else:
        bg = get_background_for_shap()
        if bg is None:
            st.warning("No hay sample_inputs.json para SHAP.")
        else:
            idx = st.number_input("Paciente ejemplo", 0, len(bg) - 1, 0)
            x_instance = bg.iloc[[idx]]

            with st.spinner("Calculando SHAP..."):
                def predict_proba_fn(x):
                    df = pd.DataFrame(x, columns=FEATURES)
                    return PIPE.predict_proba(_coerce_and_align(df))

                explainer = shap.KernelExplainer(predict_proba_fn, bg.to_numpy())
                shap_vals = explainer.shap_values(x_instance.to_numpy(), nsamples=80)

            if isinstance(shap_vals, list):
                shap_pos = shap_vals[1]
            else:
                shap_pos = shap_vals

            st.markdown("### Valores de entrada")
            st.dataframe(x_instance)

            st.markdown("### SHAP Waterfall")
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_pos[0],
                    base_values=explainer.expected_value[1]
                    if isinstance(explainer.expected_value, list)
                    else explainer.expected_value,
                    data=x_instance.to_numpy()[0],
                    feature_names=FEATURES
                ),
                show=False
            )
            st.pyplot(plt.gcf())
            plt.clf()


# ================================
# TAB 4 – DASHBOARD
# ================================
with tab_dash:
    st.subheader("📈 Dashboard de datos")

    if SAMPLE_DF is None or SAMPLE_DF.empty:
        st.warning("No hay sample_inputs.json cargado.")
    else:
        st.write("Vista previa:")
        st.dataframe(SAMPLE_DF.head())

        df_aligned = _coerce_and_align(SAMPLE_DF.copy())
        numeric_cols = [
            c for c in df_aligned.columns
            if pd.api.types.is_numeric_dtype(df_aligned[c])
        ]

        if numeric_cols:
            col = st.selectbox("Variable numérica", numeric_cols)

            fig, ax = plt.subplots()
            ax.hist(df_aligned[col].dropna(), bins=20)
            ax.set_title(f"Distribución de {col}")
            st.pyplot(fig)
        else:
            st.info("No hay columnas numéricas para graficar.")
