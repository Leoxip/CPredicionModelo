import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve,
)
import matplotlib.pyplot as plt

# ================================
# Intentamos importar SHAP (interpretabilidad) y FPDF (PDF)
# ================================
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from fpdf import FPDF
    HAS_FPDF = True
except Exception:
    HAS_FPDF = False


# ================================
# Configuración general de la app
# ================================
st.set_page_config(
    page_title="Riesgo de Preeclampsia",
    page_icon="🩺",
    layout="wide",
)

st.title("🩺 Predicción de Riesgo de Preeclampsia")
st.caption(
    "Aplicación de apoyo a la decisión clínica. No sustituye la evaluación de un profesional de la salud."
)

# ================================
# Cargar artefactos (modelo, schema, policy, samples)
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

    pipe_path = os.path.join(ART_DIR, f"pipeline_{winner_name}.joblib")
    pipe = joblib.load(pipe_path)

    sample_df = None
    if os.path.exists(sample_inputs_path):
        with open(sample_inputs_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
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


# ================================
# Funciones auxiliares
# ================================
def _coerce_and_align(df: pd.DataFrame) -> pd.DataFrame:
    for c, t in INPUT_SCHEMA.items():
        if c not in df.columns:
            df[c] = np.nan

        t_str = str(t).lower()
        if t_str.startswith(("int", "float")):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        elif t_str in ("bool", "boolean"):
            df[c] = df[c].astype("bool")
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


def style_primary_box(text, color="#0F766E"):
    st.markdown(
        f"""
        <div style="border-radius: 12px; padding: 16px; border: 1px solid {color}33;
                    background-color: {color}11; margin-bottom: 1rem;">
            {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


def generate_pdf_report(payload, prediction_dict, pdf_path="reporte_paciente.pdf"):
    if not HAS_FPDF:
        raise RuntimeError("fpdf2 no está instalado. Agrega 'fpdf2' a requirements.txt.")

    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)

    # ======== REGISTRAMOS LA FUENTE DEJAVU ==========
    font_path = "fonts/DejaVuSans.ttf"
    pdf.add_font("DejaVu", "", font_path, uni=True)
    pdf.set_font("DejaVu", "", 14)

    # ======== TÍTULO ==========
    pdf.cell(0, 10, "Reporte de Riesgo de Preeclampsia", ln=1)

    pdf.ln(4)
    pdf.set_font("DejaVu", "", 11)
    pdf.cell(0, 8, "Resultado del modelo:", ln=1)

    prob_txt = (
        f"Clasificación: {prediction_dict['pred_label']}  |  "
        f"Probabilidad: {prediction_dict['proba']*100:.2f}%"
    )

    # Usamos multi_cell porque la fuente UTF-8 lo permite
    pdf.multi_cell(0, 8, prob_txt)

    pdf.ln(4)
    pdf.cell(0, 8, "Datos ingresados:", ln=1)

    pdf.set_font("DejaVu", "", 10)

    for k, v in payload.items():
        line = f"- {k}: {v}"
        pdf.multi_cell(0, 6, line)

    pdf.output(pdf_path)
    return pdf_path



def get_background_for_shap(max_samples: int = 100):
    if SAMPLE_DF is None or SAMPLE_DF.empty:
        return None

    df = _coerce_and_align(SAMPLE_DF.copy())
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
    return df


# ================================
# Sidebar
# ================================
with st.sidebar:
    st.header("ℹ️ Información del modelo")
    st.markdown(
        f"""
        **Modelo ganador:** `{POLICY.get('winner', 'N/A')}`  
        **Umbral de decisión:** `{THRESHOLD:.2f}`
        """
    )

    if TEST_METRICS:
        st.markdown("**Métricas en test:**")
        cols = st.columns(2)
        metric_map = {
            "f1": "F1",
            "precision": "Precisión",
            "recall": "Recall",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
        }
        for i, (k, label) in enumerate(metric_map.items()):
            if k in TEST_METRICS:
                with cols[i % 2]:
                    st.metric(label, f"{TEST_METRICS[k]:.3f}")

    st.markdown("---")
    st.markdown("**Estado de librerías:**")
    st.write(f"SHAP disponible: {'✅' if HAS_SHAP else '❌'}")
    st.write(f"PDF (fpdf2) disponible: {'✅' if HAS_FPDF else '❌'}")


# ================================
# TABS
# ================================
tab_pred, tab_model, tab_shap, tab_dash, tab_report = st.tabs(
    [
        "🔮 Predicción",
        "📊 Análisis del modelo",
        "🧠 Interpretabilidad (SHAP)",
        "📈 Dashboard de datos",
        "📄 Reporte PDF",
    ]
)

# ================================
# TAB 1: Predicción
# ================================
with tab_pred:
    st.subheader("📋 Ingrese los datos clínicos de la paciente")

    with st.form("form_paciente"):
        col1, col2 = st.columns(2)

        with col1:
            edad = st.number_input("Edad (años)", min_value=10, max_value=60, value=30)
            imc = st.number_input(
                "IMC", min_value=10.0, max_value=60.0, value=25.0, step=0.1
            )
            p_sis = st.number_input(
                "Presión arterial sistólica (mmHg)",
                min_value=70,
                max_value=250,
                value=120,
            )
            p_dia = st.number_input(
                "Presión arterial diastólica (mmHg)",
                min_value=40,
                max_value=150,
                value=80,
            )

        with col2:
            hipertension = st.selectbox(
                "Antecedente de hipertensión",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
            )
            diabetes = st.selectbox(
                "Antecedente de diabetes",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
            )
            ant_fam_hiper = st.selectbox(
                "Antecedentes familiares de hipertensión",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
            )
            tec_repro_asistida = st.selectbox(
                "Uso de técnica de reproducción asistida",
                options=[0, 1],
                format_func=lambda x: "No" if x == 0 else "Sí",
            )
            creatinina = st.number_input(
                "Creatinina (mg/dL)",
                min_value=0.1,
                max_value=5.0,
                value=0.8,
                step=0.1,
            )

        submitted = st.form_submit_button("Calcular riesgo")

    prediction_dict = st.session_state.get("prediction_dict")
    payload = st.session_state.get("payload")

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

        results = predict_batch(payload)
        prediction_dict = results[0]

        st.session_state["prediction_dict"] = prediction_dict
        st.session_state["payload"] = payload

    if prediction_dict and payload:
        proba_pct = prediction_dict["proba"] * 100
        label = prediction_dict["pred_label"]

        st.markdown("---")
        st.subheader("🔍 Resultado del modelo")

        if label == "RIESGO":
            style_primary_box(
                f"<h3 style='margin:0;'>RIESGO ELEVADO</h3>"
                f"<p style='margin:0;'>Probabilidad: "
                f"<strong>{proba_pct:.2f}%</strong> (umbral = {prediction_dict['threshold']:.2f})</p>",
                color="#B91C1C",
            )
        else:
            style_primary_box(
                f"<h3 style='margin:0;'>SIN RIESGO ELEVADO</h3>"
                f"<p style='margin:0;'>Probabilidad: "
                f"<strong>{proba_pct:.2f}%</strong> (umbral = {prediction_dict['threshold']:.2f})</p>",
                color="#15803D",
            )

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Prob. riesgo", f"{proba_pct:.2f} %")
        col_b.metric("Umbral usado", f"{prediction_dict['threshold']:.2f}")
        col_c.metric("Clasificación", label)

        st.markdown("#### Datos ingresados")
        st.dataframe(pd.DataFrame([payload]))

# ================================
# TAB 2: Análisis del modelo
# ================================
with tab_model:
    st.subheader("📊 Análisis global del modelo")

    st.markdown("### 1️⃣ Métricas en test")
    if TEST_METRICS:
        st.json(TEST_METRICS)
    else:
        st.warning("No hay métricas en decision_policy.json")

    if (
        Y_TRUE is not None
        and Y_PROBA is not None
        and len(Y_TRUE) == len(Y_PROBA)
        and len(Y_TRUE) > 0
    ):
        y_pred = (Y_PROBA >= THRESHOLD).astype(int)

        st.markdown("### 2️⃣ Matriz de confusión")

        cm = confusion_matrix(Y_TRUE, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["NO RIESGO", "RIESGO"])
        ax.set_yticklabels(["NO RIESGO", "RIESGO"])
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, val, ha="center", va="center")
        fig.colorbar(im, ax=ax)
        st.pyplot(fig)

        st.markdown("### 3️⃣ ROC")
        fpr, tpr, _ = roc_curve(Y_TRUE, Y_PROBA)
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        ax2.plot([0, 1], [0, 1], "--")
        st.pyplot(fig2)

        st.markdown("### 4️⃣ Precision–Recall")
        precision, recall, _ = precision_recall_curve(Y_TRUE, Y_PROBA)
        pr_auc = auc(recall, precision)
        fig3, ax3 = plt.subplots()
        ax3.plot(recall, precision, label=f"AUC = {pr_auc:.3f}")
        st.pyplot(fig3)

# ================================
# TAB 3: SHAP
# ================================
with tab_shap:
    st.subheader("🧠 Interpretabilidad con SHAP")

    if not HAS_SHAP:
        st.warning("Instala `shap` para habilitar esta sección.")
    else:
        background_df = get_background_for_shap()
        if background_df is None:
            st.warning("No hay sample_inputs.json")
        else:
            X_bg = background_df.to_numpy()

            st.markdown("### Seleccionar paciente")
            idx = st.number_input(
                "Índice",
                min_value=0,
                max_value=len(background_df) - 1,
                value=0,
            )
            x_instance = background_df.iloc[[idx]]

            with st.spinner("Calculando SHAP..."):

                def predict_proba_fn(x):
                    df = pd.DataFrame(x, columns=FEATURES)
                    df_aligned = _coerce_and_align(df)
                    return PIPE.predict_proba(df_aligned)

                explainer = shap.KernelExplainer(predict_proba_fn, X_bg)
                shap_vals = explainer.shap_values(x_instance.to_numpy(), nsamples=100)

            if isinstance(shap_vals, list) and len(shap_vals) == 2:
                shap_positive = shap_vals[1]
            else:
                shap_positive = shap_vals

            st.markdown("### Explicación local")
            st.dataframe(x_instance)

            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_positive[0],
                    base_values=explainer.expected_value[1]
                    if isinstance(explainer.expected_value, (list, np.ndarray))
                    else explainer.expected_value,
                    data=x_instance.to_numpy()[0],
                    feature_names=FEATURES,
                ),
                show=False,
            )
            st.pyplot(plt.gcf())
            plt.clf()

# ================================
# TAB 4: Dashboard
# ================================
with tab_dash:
    st.subheader("📈 Dashboard exploratorio")

    if SAMPLE_DF is None or SAMPLE_DF.empty:
        st.warning("No hay sample_inputs.json")
    else:
        df = SAMPLE_DF.copy()
        df_aligned = _coerce_and_align(df)

        st.dataframe(df.head())

        numeric_cols = [
            c for c in df_aligned.columns if pd.api.types.is_numeric_dtype(df_aligned[c])
        ]

        if numeric_cols:
            col = st.selectbox("Variable numérica", numeric_cols)
            fig, ax = plt.subplots()
            ax.hist(df_aligned[col].dropna(), bins=20)
            st.pyplot(fig)

# ================================
# TAB 5: Reporte PDF
# ================================
with tab_report:
    st.subheader("📄 Generar reporte PDF")

    if not HAS_FPDF:
        st.warning("Necesitas instalar `fpdf2`.")
    else:
        prediction_dict = st.session_state.get("prediction_dict")
        payload = st.session_state.get("payload")

        if not prediction_dict or not payload:
            st.info("Primero realiza una predicción.")
        else:
            if st.button("📥 Generar PDF"):
                pdf_path = generate_pdf_report(payload, prediction_dict)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        "Descargar reporte",
                        data=f,
                        file_name="reporte_preeclampsia.pdf",
                        mime="application/pdf",
                    )
