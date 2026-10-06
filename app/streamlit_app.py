"""Streamlit front end for the churn model.

Run from the repo root:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.churn.config import FIG_DIR, METRICS_PATH, MODEL_PATH

st.set_page_config(page_title="Churn Radar", page_icon="📉", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .risk-card {border-radius: 14px; padding: 1.1rem 1.4rem; color: white; margin-bottom: .6rem;}
    .risk-low {background: linear-gradient(135deg,#2a9d8f,#1d6f66);}
    .risk-med {background: linear-gradient(135deg,#e9c46a,#c99a2e); color:#222;}
    .risk-high {background: linear-gradient(135deg,#e76f51,#b8432a);}
    .risk-card h2 {margin:0; font-size:2rem;}
    .risk-card p {margin:0; opacity:.9;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model():
    saved = joblib.load(MODEL_PATH)
    return saved["pipeline"], saved["columns"]


@st.cache_data
def load_metrics():
    return json.loads(METRICS_PATH.read_text())


pipe, columns = load_model()
metrics = load_metrics()

# ---------------------------------------------------------------- sidebar form
st.sidebar.title("📉 Churn Radar")
st.sidebar.caption("Describe a customer, then read the risk on the right.")

with st.sidebar:
    st.subheader("Account")
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 70.0, step=0.5)
    paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)
    payment = st.selectbox("Payment method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])

    st.subheader("Services")
    internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
    phone = st.radio("Phone service", ["Yes", "No"], horizontal=True)
    extras_on = internet != "No"
    def addon(label):
        return st.selectbox(label, ["Yes", "No"]) if extras_on else "No internet service"
    security = addon("Online security")
    backup = addon("Online backup")
    protection = addon("Device protection")
    support = addon("Tech support")
    tv = addon("Streaming TV")
    movies = addon("Streaming movies")
    lines = st.selectbox("Multiple lines", ["No", "Yes"]) if phone == "Yes" else "No phone service"

    st.subheader("Household")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior = st.radio("Senior citizen", ["No", "Yes"], horizontal=True)
    partner = st.radio("Has partner", ["No", "Yes"], horizontal=True)
    dependents = st.radio("Has dependents", ["No", "Yes"], horizontal=True)

customer = {
    "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0, "Partner": partner,
    "Dependents": dependents, "tenure": tenure, "PhoneService": phone, "MultipleLines": lines,
    "InternetService": internet, "OnlineSecurity": security, "OnlineBackup": backup,
    "DeviceProtection": protection, "TechSupport": support, "StreamingTV": tv,
    "StreamingMovies": movies, "Contract": contract, "PaperlessBilling": paperless,
    "PaymentMethod": payment, "MonthlyCharges": monthly, "TotalCharges": monthly * tenure,
}


def predict(row: dict) -> float:
    return float(pipe.predict_proba(pd.DataFrame([row])[columns])[0, 1])


prob = predict(customer)

# ---------------------------------------------------------------- main page
st.title("Customer churn risk")
tab_predict, tab_model = st.tabs(["🎯 Prediction", "📊 Model & data insights"])

with tab_predict:
    left, right = st.columns([1, 1.2], gap="large")

    with left:
        if prob < 0.30:
            tier, css, note = "Low risk", "risk-low", "Likely to stay. Keep the experience smooth."
        elif prob < 0.55:
            tier, css, note = "Medium risk", "risk-med", "Worth a check-in or a small incentive."
        else:
            tier, css, note = "High risk", "risk-high", "Prioritise for a retention offer."
        st.markdown(
            f'<div class="risk-card {css}"><p>{tier}</p><h2>{prob:.0%} churn probability</h2>'
            f"<p>{note}</p></div>", unsafe_allow_html=True)

        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=prob * 100, number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#264653"},
                   "steps": [{"range": [0, 30], "color": "#2a9d8f"},
                             {"range": [30, 55], "color": "#e9c46a"},
                             {"range": [55, 100], "color": "#e76f51"}]}))
        gauge.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=0))
        st.plotly_chart(gauge, width="stretch")

    with right:
        st.subheader("What-if: contract type")
        st.caption("Same customer, different contract. Everything else is held constant.")
        what_if = {c: predict({**customer, "Contract": c})
                   for c in ["Month-to-month", "One year", "Two year"]}
        fig = go.Figure(go.Bar(
            x=list(what_if), y=[v * 100 for v in what_if.values()],
            marker_color=["#e76f51" if k == contract else "#b0bec5" for k in what_if],
            text=[f"{v:.0%}" for v in what_if.values()], textposition="outside"))
        fig.update_layout(height=300, yaxis_title="Churn probability (%)",
                          yaxis_range=[0, 100], margin=dict(t=10, b=10))
        st.plotly_chart(fig, width="stretch")

        tips = []
        if contract == "Month-to-month":
            tips.append("Offer a discount to move to a 1- or 2-year contract.")
        if internet != "No" and security == "No":
            tips.append("Bundle online security or tech support for a trial period.")
        if payment == "Electronic check":
            tips.append("Nudge toward automatic payment, which this segment churns less on.")
        if tenure < 6:
            tips.append("New customer: schedule an onboarding check-in.")
        st.subheader("Suggested actions")
        for t in tips or ["No specific red flags. Keep monitoring."]:
            st.markdown(f"- {t}")

with tab_model:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Customers", f"{metrics['n_customers']:,}")
    c2.metric("Churn rate", f"{metrics['churn_rate']:.1%}")
    c3.metric("Test ROC-AUC", metrics["tuned_rf"]["test_auc"])
    c4.metric("Test F1 (churn)", metrics["tuned_rf"]["f1"])

    st.subheader("Model comparison")
    comp = pd.DataFrame(metrics["comparison"]).T.rename(
        columns={"cv_auc": "CV ROC-AUC", "test_auc": "Test ROC-AUC"})
    st.dataframe(comp, width="stretch")

    a, b = st.columns(2)
    a.image(str(FIG_DIR / "08_feature_importance.png"), caption="What drives the model")
    b.image(str(FIG_DIR / "02_churn_by_contract.png"), caption="Churn by contract type")
    a.image(str(FIG_DIR / "06_roc_curve.png"), caption="ROC curve")
    b.image(str(FIG_DIR / "07_confusion_matrix.png"), caption="Confusion matrix")
