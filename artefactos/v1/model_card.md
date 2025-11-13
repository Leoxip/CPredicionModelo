# Model Card — NNM
**Versión:** v1
**Sistema:** Python 3.12.12, scikit-learn 1.3.2

## Datos
Archivo: `Preeclampsia_data_clean.csv` | Shape: (1800, 9) | Objetivo: `State` (SIN RIESGO=0, RIESGO=1) | Prevalencia (RIESGO=1): 0.454

## Entrenamiento
Split 80/20 estratificado (random_state=42). Preprocesamiento: StandardScaler (num) + OneHotEncoder(ignore) (cat) + SMOTE(k=3).

## Modelo
Seleccionado para TEST: **NNM**.
Umbral de decisión: **0.50** (provisional).

## Métricas en TEST
F1=0.979, P=0.976, R=0.982,
ROC-AUC=0.999, PR-AUC=0.999.
