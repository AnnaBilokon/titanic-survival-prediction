# app.py
import os
import math
import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Titanic Survival Prediction", page_icon="🚢", layout="centered")

st.title("🚢 Titanic Survival Prediction")
st.write("Enter passenger details to estimate survival probability. "
         "This demo mirrors the feature engineering used during training.")


MODEL_PATH = "titanic_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(f"Model file not found: {MODEL_PATH}. "
             f"Please place your trained model (joblib) in the app directory.")
    st.stop()

artifact = joblib.load(MODEL_PATH)


if isinstance(artifact, dict) and "model" in artifact:
    model = artifact["model"]
    FEATURES = artifact.get("features") 
else:
    model = artifact
    FEATURES = None


DEFAULT_FEATURES = [
    "Sex", "Age", "LogFare", "FamilySize", "HasCabin",
    "Pclass_1", "Pclass_2", "Pclass_3",
]
if FEATURES is None:
    FEATURES = DEFAULT_FEATURES

with st.expander("ℹ️ Model & Features Info", expanded=False):
    st.write("**Loaded model:**", type(model).__name__)
    st.write("**Expected features (order matters):**")
    st.code(", ".join(FEATURES))


st.subheader("Passenger Inputs")

col1, col2 = st.columns(2)

with col1:
    sex = st.selectbox("Sex", ["male", "female"])
    pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3], index=2)
    age = st.slider("Age", min_value=0, max_value=80, value=30, step=1)

with col2:
    fare = st.slider("Fare", min_value=0.0, max_value=520.0, value=32.2, step=0.1)
    sibsp = st.number_input("Siblings/Spouses Aboard (SibSp)", min_value=0, max_value=10, value=0, step=1)
    parch = st.number_input("Parents/Children Aboard (Parch)", min_value=0, max_value=10, value=0, step=1)

has_cabin_bool = st.checkbox("Has Cabin listed on ticket?", value=False)


def build_feature_row(sex, age, fare, sibsp, parch, pclass, has_cabin_bool):
    family_size = int(sibsp) + int(parch) + 1
    log_fare = float(np.log1p(fare))
    sex_num = 1 if sex == "female" else 0
    has_cabin = 1 if has_cabin_bool else 0

    row = {
        "Sex": sex_num,
        "Age": float(age),
        "LogFare": log_fare,
        "FamilySize": family_size,
        "HasCabin": has_cabin,
    
        "Pclass_1": 1 if pclass == 1 else 0,
        "Pclass_2": 1 if pclass == 2 else 0,
        "Pclass_3": 1 if pclass == 3 else 0,
    }
    return row

row = build_feature_row(sex, age, fare, sibsp, parch, pclass, has_cabin_bool)


def make_X_input(row, feature_order):

    X = pd.DataFrame([row])

    for col in feature_order:
        if col not in X.columns:
            X[col] = 0


    X = X[feature_order]
    return X

X_input = make_X_input(row, FEATURES)


st.subheader("Prediction")

def predict_with_model(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
        pred = (proba >= 0.5).astype(int)
        score = float(proba[0])
    elif hasattr(model, "decision_function"):

        s = model.decision_function(X)
        s_min, s_max = float(np.min(s)), float(np.max(s))
        score = (s - s_min) / (s_max - s_min + 1e-12)
        pred = (score >= 0.5).astype(int)
        score = float(score[0])
    else:
        pred = model.predict(X)
        score = float(pred[0])  
    return int(pred[0]), score

if st.button("Predict Survival"):
    try:
        pred, prob = predict_with_model(model, X_input)
        st.success("✅ Survived" if pred == 1 else "❌ Did not survive")
        st.metric("Estimated Survival Probability", f"{prob:.2%}")


    except Exception as e:
        st.error(f"Prediction failed: {e}")


st.markdown("---")
st.caption(
    "Trained on Titanic features: Sex, Age, LogFare, FamilySize, HasCabin, and one-hot Pclass. "
)
