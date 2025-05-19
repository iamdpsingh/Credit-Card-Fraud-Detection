import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

# Load model if exists, else show error
def load_model(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"Model file not found at {path}. Please train the model first.")
        return None

# Load scaler if exists, else show error
def load_scaler(path):
    if os.path.exists(path):
        return joblib.load(path)
    else:
        st.error(f"Scaler file not found at {path}. Please train the model first.")
        return None

st.title("💳 Credit Card Fraud Detection")

uploaded_file = st.file_uploader("Upload creditcard.csv", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Data Preview:", df.head())

    if st.button("Train Model"):
        # Data Preprocessing
        scaler = StandardScaler()
        df['normalizedAmount'] = scaler.fit_transform(df['Amount'].values.reshape(-1, 1))
        df.drop(['Time', 'Amount'], axis=1, inplace=True)

        X = df.drop('Class', axis=1)
        y = df['Class']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X_train, y_train)

        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_res, y_res)

        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        st.success("Model Trained Successfully!")
        st.text("Classification Report:\n" + classification_report(y_test, y_pred))
        st.write("ROC-AUC Score:", roc_auc_score(y_test, y_prob))

        # Create directory if not exists
        os.makedirs("models", exist_ok=True)
        # Save model and scaler
        joblib.dump(model, "models/best_model.pkl")
        joblib.dump(scaler, "models/scaler.pkl")

if st.button("Predict on Sample"):
    if uploaded_file is None:
        st.error("Please upload a CSV file first.")
    else:
        model = load_model("models/best_model.pkl")
        scaler = load_scaler("models/scaler.pkl")

        if model is not None and scaler is not None:
            # Sample from original df with all columns
            sample = df.sample(1).copy()

            # Preprocess sample same as training
            sample['normalizedAmount'] = scaler.transform(sample['Amount'].values.reshape(-1, 1))
            sample.drop(['Time', 'Amount', 'Class'], axis=1, inplace=True)

            pred = model.predict(sample)
            prob = model.predict_proba(sample)[0][1]

            st.write("Prediction:", "Fraud" if pred[0] == 1 else "Not Fraud")
            st.write("Probability of Fraud:", round(prob * 100, 2), "%")
