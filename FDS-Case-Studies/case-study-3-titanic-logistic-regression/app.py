"""
Streamlit app: Titanic survival predictor (Case Study 3).

Loads the fitted scikit-learn pipeline saved by the notebook
(preprocessing + logistic regression) and predicts survival probability
for a passenger profile entered by the user.

Run locally:   streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "model" / "titanic_logreg_pipeline.joblib"
FEATURES = ["Age", "SibSp", "Parch", "LogFare", "Pclass", "Sex", "Embarked", "HasCabin", "AgeMissing"]
PORTS = {"Southampton (S)": "S", "Cherbourg (C)": "C", "Queenstown (Q)": "Q"}

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢", layout="centered")


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("🚢 Titanic Survival Predictor")
st.caption("Logistic regression trained on 712 passengers · test accuracy 79.3% · ROC-AUC 0.833")

if model is None:
    st.error(f"Model file not found at `{MODEL_PATH}`. Run the notebook first to create it.")
    st.stop()

with st.form("passenger"):
    c1, c2 = st.columns(2)
    with c1:
        sex = st.radio("Sex", ["female", "male"], horizontal=True)
        pclass = st.selectbox("Passenger class", [1, 2, 3], index=2,
                              format_func=lambda c: {1: "1st", 2: "2nd", 3: "3rd"}[c])
        age_known = st.checkbox("Age known", value=True)
        age = st.slider("Age (years)", 0.5, 80.0, 28.0, 0.5, disabled=not age_known)
        port = st.selectbox("Port of embarkation", list(PORTS))
    with c2:
        sibsp = st.number_input("Siblings / spouses aboard", 0, 8, 0)
        parch = st.number_input("Parents / children aboard", 0, 6, 0)
        fare = st.number_input("Fare paid (£)", 0.0, 520.0, 14.45, 0.5,
                               help="Median fare was £14.45; 1st class typically £30–£500.")
        has_cabin = st.checkbox("Cabin number recorded", value=(pclass == 1))
    submitted = st.form_submit_button("Predict survival", use_container_width=True)

if submitted:
    row = pd.DataFrame([{
        # A missing age is passed as NaN so the pipeline's median imputer fills it,
        # exactly as it did for passengers with unknown age during training.
        "Age": age if age_known else np.nan,
        "SibSp": int(sibsp),
        "Parch": int(parch),
        "LogFare": float(np.log1p(fare)),
        "Pclass": int(pclass),
        "Sex": sex,
        "Embarked": PORTS[port],
        "HasCabin": int(has_cabin),
        "AgeMissing": int(not age_known),
    }])[FEATURES]

    p = float(model.predict_proba(row)[0, 1])
    survived = p >= 0.5

    st.subheader("Prediction")
    m1, m2 = st.columns(2)
    m1.metric("Survival probability", f"{p:.1%}")
    m2.metric("Predicted outcome", "Survived" if survived else "Did not survive")
    st.progress(p)

    if survived:
        st.success("The model predicts this passenger would **survive** (probability ≥ 50%).")
    else:
        st.error("The model predicts this passenger would **not survive** (probability < 50%).")

    st.caption(
        "Strongest drivers in this model: sex (male ≈ 12× lower odds), passenger class / cabin / fare, "
        "age (older → lower odds) and siblings/spouses aboard. This is a statistical estimate from "
        "historical data, not a certainty."
    )

with st.expander("About the model"):
    st.markdown(
        "- **Algorithm:** logistic regression (scikit-learn, L2-regularised, C = 1.0)\n"
        "- **Inputs:** sex, class, age (median-imputed if unknown), siblings/spouses, parents/children, "
        "log(1 + fare), port, cabin-recorded flag\n"
        "- **Test-set performance:** accuracy 0.793 · precision 0.767 · recall 0.667 · F1 0.713 · ROC-AUC 0.833"
    )
