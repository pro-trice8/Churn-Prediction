import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.predict import load_model
from src.features import build_features

st.set_page_config(page_title="Churn Predictor", page_icon="📊", layout="wide")

model, columns = load_model()


def add_sql_features(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(df["MonthlyCharges"])
    df["charges_per_tenure"] = (df["TotalCharges"] / df["tenure"].replace(0, 1)).round(2)
    df["tenure_group"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, 100],
        labels=["0-1yr", "1-2yr", "2-4yr", "4-6yr"]
    ).astype(str)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    return df


def encode_row(row_df):
    encoded = build_features(add_sql_features(row_df)).drop(columns=["Churn"], errors="ignore")
    return encoded.reindex(columns=columns, fill_value=0)


def gauge(prob):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 40, "color": "#E4E4E7"}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#71717A"},
            "bar": {"color": "#4F46E5"},
            "bgcolor": "#1A1D29",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 35], "color": "#14532D"},
                {"range": [35, 65], "color": "#713F12"},
                {"range": [65, 100], "color": "#7F1D1D"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 3},
                "thickness": 0.8,
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        height=260, margin=dict(l=20, r=20, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)", font={"color": "#E4E4E7"}
    )
    return fig


# ---------- Custom CSS for the result card ----------
st.markdown("""
<style>
.result-card {
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    margin-top: 0.5rem;
    font-size: 1.05rem;
}
.high-risk { background: #2A1215; border: 1px solid #7F1D1D; color: #FCA5A5; }
.low-risk  { background: #0F2417; border: 1px solid #14532D; color: #86EFAC; }
.app-subtitle { color: #A1A1AA; font-size: 0.95rem; margin-top: -0.6rem; }
</style>
""", unsafe_allow_html=True)

st.title("Customer Churn Prediction")
st.markdown('<p class="app-subtitle">Predict churn risk and understand the drivers behind each prediction.</p>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Single Customer", "Batch Upload"])

# ================= TAB 1 =================
with tab1:
    # ---- Inputs live in the sidebar ----
    with st.sidebar:
        st.header("Customer details")

        st.subheader("Account")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        Contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check",
            "Bank transfer (automatic)", "Credit card (automatic)"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])

        st.subheader("Charges")
        MonthlyCharges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
        TotalCharges = st.number_input("Total Charges", 0.0, 9000.0, 800.0)

        st.subheader("Services")
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])

        st.subheader("Demographics")
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])

        predict_btn = st.button("Predict Churn", type="primary", use_container_width=True)

    # ---- Results in the main panel ----
    if predict_btn:
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner,
            "Dependents": Dependents, "tenure": tenure, "PhoneService": PhoneService,
            "MultipleLines": "No", "InternetService": InternetService,
            "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No",
            "TechSupport": "No", "StreamingTV": "No", "StreamingMovies": "No",
            "Contract": Contract, "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges,
            "TotalCharges": TotalCharges, "Churn": "No",
        }])

        encoded = encode_row(row)
        p = model.predict_proba(encoded)[:, 1][0]

        # Top metrics row
        c1, c2, c3 = st.columns(3)
        c1.metric("Churn Probability", f"{p:.1%}")
        c2.metric("Risk Level", "High" if p > 0.5 else "Low")
        c3.metric("Recommended Action",
                  "Retention offer" if p > 0.5 else "Monitor")

        left, right = st.columns([1, 1])

        with left:
            st.plotly_chart(gauge(p), use_container_width=True)

        with right:
            if p > 0.5:
                st.markdown(
                    '<div class="result-card high-risk"><b>High risk.</b> '
                    'This customer shows strong churn signals. Consider a '
                    'targeted retention offer.</div>', unsafe_allow_html=True)
            else:
                st.markdown(
                    '<div class="result-card low-risk"><b>Low risk.</b> '
                    'This customer is likely to stay. Routine monitoring '
                    'is sufficient.</div>', unsafe_allow_html=True)

        # SHAP explanation
        st.markdown("### Why this prediction?")
        st.caption("Features pushing toward churn (red) or retention (green) for this customer.")

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(encoded)

        contributions = pd.DataFrame({
            "feature": columns, "impact": shap_values[0]
        })
        contributions["abs"] = contributions["impact"].abs()
        top = contributions.sort_values("abs", ascending=False).head(8)

        fig, ax = plt.subplots(figsize=(8, 4))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        bar_colors = ["#EF4444" if v > 0 else "#22C55E" for v in top["impact"]]
        ax.barh(top["feature"], top["impact"], color=bar_colors)
        ax.invert_yaxis()
        ax.tick_params(colors="#A1A1AA")
        ax.set_xlabel("Impact on churn risk", color="#A1A1AA")
        for spine in ax.spines.values():
            spine.set_color("#3F3F46")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Fill in the customer details in the sidebar and click **Predict Churn**.")

# ================= TAB 2 =================
with tab2:
    st.subheader("Score many customers at once")
    st.caption("Upload a CSV with the same columns as the Telco dataset.")
    file = st.file_uploader("CSV file", type="csv")
    if file:
        df = pd.read_csv(file)
        encoded = build_features(add_sql_features(df)).drop(columns=["Churn"], errors="ignore")
        encoded = encoded.reindex(columns=columns, fill_value=0)
        df["churn_probability"] = model.predict_proba(encoded)[:, 1]
        df["risk"] = (df["churn_probability"] > 0.5).map({True: "High", False: "Low"})

        high = (df["risk"] == "High").sum()
        m1, m2, m3 = st.columns(3)
        m1.metric("Customers scored", len(df))
        m2.metric("High risk", high)
        m3.metric("High-risk rate", f"{high/len(df):.1%}")

        st.dataframe(
            df.sort_values("churn_probability", ascending=False),
            use_container_width=True
        )
        st.download_button("Download results", df.to_csv(index=False),
                           "scored_customers.csv")