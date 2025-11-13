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

# Intentamos importar shap (interpretabilidad) y reportlab (PDF)
try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    HAS_REPORTLAB = True
except Exception:
    HAS_REPORTLAB = False


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
    # Rutas
    input_schema_path = os.path.join(ART_DIR, "input_schema.json")
    label_map_path = os.path.join(ART_DIR, "label_map.json")
    policy_path = os.path.join(ART_DIR, "decision_policy.json")
    sample_inputs_path = os.path.join(ART_DIR, "sample_inputs.json")

    # JSONs básicos
    with open(input_schema_path, "r", encoding="utf-8") as f:
        input_schema = json.load(f)
    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map = json.load(f)
    with open(policy_path, "r", encoding="utf-8") as f:
        policy = json.load(f)

    winner_name = policy["winner"]
    threshold = float(policy.get("threshold", 0.5))

    # Pipeline ganador
    pipe_path = os.path.join(ART_DIR, f"pipeline_{winner_name}.joblib")
    pipe = joblib.load(pipe_path)

    # Sample inputs (para interpretabilidad / dashboard)
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
    """Asegura tipos según INPUT_SCHEMA y alinea columnas en el orden esperado."""
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
    """
    records: dict o lista de dicts con las features de entrada.
    thr: umbral opcional, si no se pasa se usa THRESHOLD.
    """
    thr = THRESHOLD if thr is None else float(thr)

    if isinstance(records, dict):
        records = [records]

    df = _coerce_and_align(pd.DataFrame(records))
    proba = PIPE.predict_proba(df)[:, 1]  # Prob(RIESGO=1 | x)
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
    if not HAS_REPORTLAB:
        raise RuntimeError(
            "ReportLab no está instalado. Agrega 'reportlab' a requirements.txt"
        )

    c = canvas.Canvas(pdf_path, pagesize=A4)
    width, height = A4

    line_y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, line_y, "Reporte de Riesgo de Preeclampsia")
    line_y -= 30

    c.setFont("Helvetica", 11)
    c.drawString(50, line_y, "Resultado del modelo:")
    line_y -= 20
    c.drawString(
        70,
        line_y,
        f"Clasificación: {prediction_dict['pred_label']}  |  Probabilidad: {prediction_dict['proba']*100:.2f}%",
    )
    line_y -= 30

    c.setFont("Helvetica", 11)
    c.drawString(50, line_y, "Datos ingresados:")
    line_y -= 20
    for k, v in payload.items():
        c.drawString(70, line_y, f"- {k}: {v}")
        line_y -= 16
        if line_y < 80:
            c.showPage()
            line_y = height - 50

    c.showPage()
    c.save()
    return pdf_path


def get_background_for_shap(max_samples: int = 100):
    """Devuelve un background pequeño para SHAP usando SAMPLE_DF."""
    if SAMPLE_DF is None or SAMPLE_DF.empty:
        return None

    df = _coerce_and_align(SAMPLE_DF.copy())
    if len(df) > max_samples:
        df = df.sample(max_samples, random_state=42)
    return df


# ================================
# Sidebar: info del modelo
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
    st.write(f"ReportLab disponible: {'✅' if HAS_REPORTLAB else '❌'}")


# ================================
# Tabs principales (múltiples vistas)
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

    # Hacemos estas variables visibles globalmente para usarlas en el tab Reporte PDF
    prediction_dict = st.session_state.get("prediction_dict")
    payload = st.session_state.get("payload")

    if submitted:
        warnings = []
        if p_sis < 80 or p_sis > 200:
            warnings.append(
                "La presión sistólica está fuera del rango típico clínico (≈80–200)."
            )
        if p_dia < 40 or p_dia > 120:
            warnings.append(
                "La presión diastólica está fuera del rango típico clínico (≈40–120)."
            )

        if warnings:
            for w in warnings:
                st.warning(w)

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

        # guardamos en sesión para otras pestañas
        st.session_state["prediction_dict"] = prediction_dict
        st.session_state["payload"] = payload

    if prediction_dict is not None and payload is not None:
        proba_pct = prediction_dict["proba"] * 100
        label = prediction_dict["pred_label"]

        st.markdown("---")
        st.subheader("🔍 Resultado del modelo")

        if label == "RIESGO":
            style_primary_box(
                f"<h3 style='margin:0;'>RIESGO ELEVADO</h3>"
                f"<p style='margin:0;'>Probabilidad estimada de riesgo: "
                f"<strong>{proba_pct:.2f}%</strong> (umbral = {prediction_dict['threshold']:.2f})</p>",
                color="#B91C1C",
            )
        else:
            style_primary_box(
                f"<h3 style='margin:0;'>SIN RIESGO ELEVADO</h3>"
                f"<p style='margin:0;'>Probabilidad estimada de riesgo: "
                f"<strong>{proba_pct:.2f}%</strong> (umbral = {prediction_dict['threshold']:.2f})</p>",
                color="#15803D",
            )

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Prob. riesgo", f"{proba_pct:.2f} %")
        with col_b:
            st.metric("Umbral usado", f"{prediction_dict['threshold']:.2f}")
        with col_c:
            st.metric("Clasificación", label)

        st.markdown("#### Datos ingresados")
        st.dataframe(pd.DataFrame([payload]))

        st.info(
            "Este resultado debe interpretarse siempre junto con la historia clínica "
            "y la evaluación de un profesional de la salud."
        )

# ================================
# TAB 2: Análisis del modelo
# ================================
with tab_model:
    st.subheader("📊 Análisis global del modelo")

    st.markdown("### 1️⃣ Métricas en conjunto de test")
    if TEST_METRICS:
        st.json(TEST_METRICS)
        st.info(
            "- **F1** combina precisión y recall.\n"
            "- **Precisión**: de los casos que el modelo marcó como riesgo, cuántos realmente lo son.\n"
            "- **Recall**: de los casos de riesgo reales, cuántos detecta el modelo.\n"
            "- **ROC-AUC** y **PR-AUC** resumen el desempeño en diferentes umbrales."
        )
    else:
        st.warning("No se encontraron métricas de test en decision_policy.json.")

    if (
        Y_TRUE is not None
        and Y_PROBA is not None
        and len(Y_TRUE) == len(Y_PROBA)
        and len(Y_TRUE) > 0
    ):
        y_pred = (Y_PROBA >= THRESHOLD).astype(int)

        # Matriz de confusión
        st.markdown("### 2️⃣ Matriz de confusión")

        cm = confusion_matrix(Y_TRUE, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Valor real")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["NO RIESGO", "RIESGO"])
        ax.set_yticklabels(["NO RIESGO", "RIESGO"])

        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, val, ha="center", va="center")

        fig.colorbar(im, ax=ax)
        st.pyplot(fig)
        st.caption("La matriz de confusión muestra aciertos y errores del modelo en cada clase.")

        # Curva ROC
        st.markdown("### 3️⃣ Curva ROC")
        fpr, tpr, _ = roc_curve(Y_TRUE, Y_PROBA)
        roc_auc = auc(fpr, tpr)

        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
        ax2.plot([0, 1], [0, 1], "--", label="Azar")
        ax2.set_xlabel("Tasa de falsos positivos (FPR)")
        ax2.set_ylabel("Tasa de verdaderos positivos (TPR)")
        ax2.legend()
        st.pyplot(fig2)
        st.caption("Cuanto más se acerque la curva a la esquina superior izquierda, mejor es el modelo.")

        # Curva Precision-Recall
        st.markdown("### 4️⃣ Curva Precision–Recall")
        precision, recall, _ = precision_recall_curve(Y_TRUE, Y_PROBA)
        pr_auc = auc(recall, precision)

        fig3, ax3 = plt.subplots()
        ax3.plot(recall, precision, label=f"PR (AUC = {pr_auc:.3f})")
        ax3.set_xlabel("Recall")
        ax3.set_ylabel("Precision")
        ax3.legend()
        st.pyplot(fig3)
        st.caption(
            "Especialmente útil cuando la clase positiva es poco frecuente: "
            "muestra el equilibrio entre detectar casos (recall) y evitar falsos positivos (precision)."
        )

        # Importancia de características (si el modelo la soporta)
        st.markdown("### 5️⃣ Importancia de características (si aplica)")
        model = None
        if hasattr(PIPE, "named_steps"):
            model = PIPE.named_steps.get("model", None)

        if model is not None and hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            idx = np.argsort(importances)

            fig4, ax4 = plt.subplots()
            ax4.barh(np.array(FEATURES)[idx], importances[idx])
            ax4.set_xlabel("Importancia")
            ax4.set_title("Importancia de variables para el modelo")
            st.pyplot(fig4)
            st.caption("Estas variables aportan más información al modelo para decidir el riesgo.")
        else:
            st.info(
                "El estimador final del pipeline no expone `feature_importances_` "
                "(por ejemplo, k-NN o modelos basados en distancias). "
                "Usa la pestaña de *Interpretabilidad (SHAP)* para un análisis más detallado."
            )
    else:
        st.warning(
            "Para mostrar matriz de confusión y curvas ROC/PR necesitas guardar `y_true` y "
            "`y_pred_proba` en decision_policy.json."
        )

    st.markdown("### 6️⃣ Información del pipeline")
    st.write(PIPE)

# ================================
# TAB 3: Interpretabilidad (SHAP)
# ================================
with tab_shap:
    st.subheader("🧠 Interpretabilidad con SHAP (modelo-agnóstico)")

    if not HAS_SHAP:
        st.warning(
            "SHAP no está instalado. Agrega `shap` a tu requirements.txt para habilitar esta sección."
        )
    else:
        background_df = get_background_for_shap()
        if background_df is None:
            st.warning(
                "No se encontró SAMPLE_DF a partir de sample_inputs.json. "
                "Agrega ejemplos de pacientes para poder calcular explicaciones SHAP."
            )
        else:
            st.markdown(
                "Usamos **SHAP KernelExplainer**, que funciona con cualquier modelo "
                "(incluyendo k-NN u otros basados en distancias)."
            )

            X_bg = background_df.to_numpy()

            # Seleccionar un paciente para explicación local
            st.markdown("#### 1️⃣ Seleccione un paciente de ejemplo")
            idx = st.number_input(
                "Índice del paciente (de sample_inputs.json)",
                min_value=0,
                max_value=len(background_df) - 1,
                value=0,
            )
            x_instance = background_df.iloc[[idx]]

            with st.spinner("Calculando valores SHAP (puede tardar unos segundos)..."):

                # KernelExplainer requiere una función que devuelva probabilidades
                def predict_proba_fn(x):
                    df = pd.DataFrame(x, columns=FEATURES)
                    df_aligned = _coerce_and_align(df)
                    return PIPE.predict_proba(df_aligned)

                explainer = shap.KernelExplainer(predict_proba_fn, X_bg)
                shap_vals = explainer.shap_values(x_instance.to_numpy(), nsamples=100)

            # Para binario, shap_vals es una lista [clase0, clase1]; usamos clase positiva
            if isinstance(shap_vals, list) and len(shap_vals) == 2:
                shap_positive = shap_vals[1]
            else:
                shap_positive = shap_vals

            st.markdown("#### 2️⃣ Explicación local (para la paciente seleccionada)")
            st.write("Valores de entrada:")
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
            fig_local = plt.gcf()
            st.pyplot(fig_local)
            plt.clf()

            st.markdown("#### 3️⃣ Importancia global (summary plot)")
            shap.summary_plot(
                shap_positive,
                background_df,
                feature_names=FEATURES,
                show=False,
            )
            fig_sum = plt.gcf()
            st.pyplot(fig_sum)
            plt.clf()

# ================================
# TAB 4: Dashboard de datos
# ================================
with tab_dash:
    st.subheader("📈 Dashboard exploratorio de datos")

    if SAMPLE_DF is None or SAMPLE_DF.empty:
        st.warning(
            "No se encontró sample_inputs.json con datos de ejemplo. "
            "Agrega este archivo para habilitar el dashboard."
        )
    else:
        df = SAMPLE_DF.copy()
        df_aligned = _coerce_and_align(df)

        st.markdown("### 1️⃣ Vista general de los datos de ejemplo")
        st.dataframe(df.head())

        st.markdown("### 2️⃣ Distribución de variables numéricas")
        numeric_cols = [
            c for c in df_aligned.columns if pd.api.types.is_numeric_dtype(df_aligned[c])
        ]

        if numeric_cols:
            col_var = st.selectbox("Selecciona variable numérica", numeric_cols)
            fig_hist, ax_hist = plt.subplots()
            ax_hist.hist(df_aligned[col_var].dropna(), bins=20)
            ax_hist.set_title(f"Histograma de {col_var}")
            st.pyplot(fig_hist)
        else:
            st.info("No se detectaron variables numéricas para graficar.")

        st.markdown("### 3️⃣ Correlación entre variables numéricas")
        if len(numeric_cols) >= 2:
            corr = df_aligned[numeric_cols].corr()
            fig_corr, ax_corr = plt.subplots()
            im = ax_corr.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
            ax_corr.set_xticks(range(len(numeric_cols)))
            ax_corr.set_yticks(range(len(numeric_cols)))
            ax_corr.set_xticklabels(numeric_cols, rotation=90)
            ax_corr.set_yticklabels(numeric_cols)
            fig_corr.colorbar(im, ax=ax_corr)
            st.pyplot(fig_corr)
        else:
            st.info(
                "Se necesitan al menos dos variables numéricas para la matriz de correlación."
            )

# ================================
# TAB 5: Reporte PDF
# ================================
with tab_report:
    st.subheader("📄 Generar reporte en PDF")

    st.markdown(
        "El reporte incluye el resultado del modelo y los datos ingresados en la pestaña de **Predicción**."
    )

    if not HAS_REPORTLAB:
        st.warning(
            "Para generar PDF necesitas instalar `reportlab` y añadirlo a tu requirements.txt."
        )
    else:
        prediction_dict = st.session_state.get("prediction_dict")
        payload = st.session_state.get("payload")

        if not prediction_dict or not payload:
            st.info(
                "Primero realiza una predicción en la pestaña **Predicción**. "
                "Luego vuelve aquí para descargar el reporte."
            )
        else:
            if st.button("📥 Generar y descargar reporte en PDF"):
                pdf_path = generate_pdf_report(payload, prediction_dict)
                with open(pdf_path, "rb") as f:
                    st.download_button(
                        label="Descargar reporte",
                        data=f,
                        file_name="reporte_preeclampsia.pdf",
                        mime="application/pdf",
                    )
