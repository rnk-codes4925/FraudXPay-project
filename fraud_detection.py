import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
import shap
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FraudXPay — Fraud Detection",
    page_icon="🛡️",
    layout="wide",
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title  { font-size:2.2rem; font-weight:700; color:#1a1a2e; }
    .sub-title   { font-size:1rem; color:#555; margin-bottom:1.5rem; }
    .risk-low    { background:#d4edda; color:#155724; padding:12px 20px;
                   border-radius:8px; font-weight:600; font-size:1.1rem; }
    .risk-medium { background:#fff3cd; color:#856404; padding:12px 20px;
                   border-radius:8px; font-weight:600; font-size:1.1rem; }
    .risk-high   { background:#ffe0b2; color:#7f3f00; padding:12px 20px;
                   border-radius:8px; font-weight:600; font-size:1.1rem; }
    .risk-critical{ background:#f8d7da; color:#721c24; padding:12px 20px;
                    border-radius:8px; font-weight:600; font-size:1.1rem; }
    .explain-box { background:#f8f9fa; border-left:4px solid #4a90d9;
                   padding:14px 18px; border-radius:4px; margin-top:10px; }
    .metric-card { background:#ffffff; border:1px solid #e0e0e0;
                   border-radius:10px; padding:16px; text-align:center; }
</style>
""", unsafe_allow_html=True)


# ─── LOAD MODEL ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("fraud_detection_pipeline.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()


# ─── HELPER: get classifier from pipeline ───────────────────────────────────────
def get_classifier(pipeline):
    """Extract the final estimator from a sklearn Pipeline."""
    from sklearn.pipeline import Pipeline
    if hasattr(pipeline, "steps"):
        return pipeline.steps[-1][1]
    return pipeline


# ─── HELPER: risk label ─────────────────────────────────────────────────────────
def risk_label(score):
    if score < 25:
        return "✅ Low Risk", "risk-low", "#27ae60"
    elif score < 50:
        return "⚠️ Medium Risk", "risk-medium", "#f39c12"
    elif score < 75:
        return "🔶 High Risk", "risk-high", "#e67e22"
    else:
        return "🚨 Critical Risk — Block Transaction", "risk-critical", "#e74c3c"


# ─── SIDEBAR INPUTS ─────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/000000/security-shield-green.png", width=80)
st.sidebar.markdown("## 🏦 Transaction Details")

tx_type = st.sidebar.selectbox(
    "Transaction Type",
    ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT", "CASH_IN"],
    help="Type of the transaction"
)

amount = st.sidebar.number_input(
    "Transaction Amount (₹)",
    min_value=0.0, max_value=10_000_000.0,
    value=50_000.0, step=1000.0
)

st.sidebar.markdown("**Sender Account**")
old_balance_orig = st.sidebar.number_input(
    "Balance Before Transaction (₹)",
    min_value=0.0, max_value=10_000_000.0,
    value=80_000.0, step=1000.0
)
new_balance_orig = st.sidebar.number_input(
    "Balance After Transaction (₹)",
    min_value=0.0, max_value=10_000_000.0,
    value=30_000.0, step=1000.0
)

st.sidebar.markdown("**Receiver Account**")
old_balance_dest = st.sidebar.number_input(
    "Receiver Balance Before (₹)",
    min_value=0.0, max_value=10_000_000.0,
    value=1_000.0, step=1000.0
)
new_balance_dest = st.sidebar.number_input(
    "Receiver Balance After (₹)",
    min_value=0.0, max_value=10_000_000.0,
    value=51_000.0, step=1000.0
)

step = st.sidebar.slider("Transaction Hour (step)", 1, 744, 1,
                          help="Hour of the simulation (1–744)")

analyze_btn = st.sidebar.button("🔍 Analyze Transaction", use_container_width=True)


# ─── MAIN HEADER ────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">🛡️ FraudXPay — Fraud Detection</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-powered real-time fraud detection with Risk Score & Explainable AI</p>',
            unsafe_allow_html=True)

# ─── ENCODE TYPE ────────────────────────────────────────────────────────────────
type_map = {"TRANSFER": 4, "CASH_OUT": 1, "PAYMENT": 3, "DEBIT": 2, "CASH_IN": 0}
type_encoded = type_map[tx_type]


# ─── BUILD INPUT DATAFRAME ──────────────────────────────────────────────────────
# Derived features — these help the model a lot
balance_drain_ratio   = amount / (old_balance_orig + 1)
sender_mismatch       = abs((old_balance_orig - new_balance_orig) - amount)
receiver_mismatch     = abs((new_balance_dest - old_balance_dest) - amount)

input_data = pd.DataFrame([{
    "step":             step,
    "type":             type_encoded,
    "amount":           amount,
    "oldbalanceOrg":    old_balance_orig,
    "newbalanceOrig":   new_balance_orig,
    "oldbalanceDest":   old_balance_dest,
    "newbalanceDest":   new_balance_dest,
    "balance_drain_ratio": balance_drain_ratio,
    "sender_mismatch":     sender_mismatch,
    "receiver_mismatch":   receiver_mismatch,
}])

# Keep only columns the model was trained on
try:
    expected_cols = model.feature_names_in_  # sklearn >= 1.0
    input_data = input_data[[c for c in expected_cols if c in input_data.columns]]
except AttributeError:
    pass


# ─── PREDICTION & RESULTS ───────────────────────────────────────────────────────
if analyze_btn:
    with st.spinner("Analyzing transaction..."):

        # Prediction
        prediction   = model.predict(input_data)[0]
        try:
            fraud_prob = model.predict_proba(input_data)[0][1]
        except Exception:
            fraud_prob = float(prediction)

        risk_score = int(round(fraud_prob * 100))
        label, css_class, color = risk_label(risk_score)

        # ── TOP METRICS ─────────────────────────────────────────────────────────
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Risk Score",       f"{risk_score} / 100")
        col2.metric("Fraud Probability", f"{fraud_prob:.1%}")
        col3.metric("Verdict",           "🚨 FRAUD" if prediction == 1 else "✅ LEGIT")
        col4.metric("Amount",            f"₹{amount:,.0f}")

        st.markdown("---")

        # ── RISK SCORE GAUGE ────────────────────────────────────────────────────
        left, right = st.columns([1, 1])

        with left:
            st.subheader("📊 Risk Score Gauge")
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                delta={"reference": 50, "valueformat": ".0f"},
                title={"text": "Fraud Risk Score (0 = Safe · 100 = Fraud)"},
                number={"suffix": " / 100", "font": {"size": 28}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1},
                    "bar":  {"color": color, "thickness": 0.25},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "steps": [
                        {"range": [0, 25],   "color": "#d4edda"},
                        {"range": [25, 50],  "color": "#fff3cd"},
                        {"range": [50, 75],  "color": "#ffe0b2"},
                        {"range": [75, 100], "color": "#f8d7da"},
                    ],
                    "threshold": {
                        "line":      {"color": color, "width": 5},
                        "thickness": 0.8,
                        "value":     risk_score
                    }
                }
            ))
            gauge.update_layout(height=300, margin=dict(t=40, b=0, l=20, r=20))
            st.plotly_chart(gauge, use_container_width=True)

        with right:
            st.subheader("🏷️ Verdict")
            st.markdown(f'<div class="{css_class}">{label}</div>', unsafe_allow_html=True)
            st.markdown("")

            # Recommendation
            if risk_score < 25:
                st.info("✅ **Recommendation:** Transaction can proceed normally. No action required.")
            elif risk_score < 50:
                st.warning("⚠️ **Recommendation:** Apply 2-factor authentication before approving.")
            elif risk_score < 75:
                st.error("🔶 **Recommendation:** Flag for manual review by compliance team.")
            else:
                st.error("🚨 **Recommendation:** Block transaction immediately and notify the account holder.")

            # Probability bar
            st.markdown("**Probability breakdown:**")
            prob_df = pd.DataFrame({
                "Category":    ["Legitimate", "Fraudulent"],
                "Probability": [round(1 - fraud_prob, 4), round(fraud_prob, 4)]
            })
            fig_prob = px.bar(prob_df, x="Category", y="Probability",
                              color="Category",
                              color_discrete_map={"Legitimate": "#27ae60", "Fraudulent": "#e74c3c"},
                              range_y=[0, 1])
            fig_prob.update_layout(showlegend=False, height=220,
                                   margin=dict(t=10, b=10, l=10, r=10))
            st.plotly_chart(fig_prob, use_container_width=True)

        st.markdown("---")

        # ── EXPLAINABLE AI — SHAP ───────────────────────────────────────────────
        st.subheader("🧠 Explainable AI — Why did the model decide this?")
        st.caption("Red bars = features that **increase** fraud risk. "
                   "Blue bars = features that **decrease** fraud risk.")

        try:
            clf = get_classifier(model)
            # Get transformed data if pipeline has preprocessor
            try:
                from sklearn.pipeline import Pipeline
                if isinstance(model, Pipeline) and len(model.steps) > 1:
                    X_transformed = model[:-1].transform(input_data)
                else:
                    X_transformed = input_data.values
            except Exception:
                X_transformed = input_data.values

            explainer   = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_transformed)

            # For binary classifiers shap_values is a list → pick fraud class (index 1)
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            else:
                sv = shap_values[0]

            feat_names = list(input_data.columns)
            shap_df = pd.DataFrame({
                "Feature":    feat_names[:len(sv)],
                "SHAP Value": sv[:len(feat_names)],
            }).sort_values("SHAP Value", key=abs, ascending=False)

            shap_df["Direction"] = shap_df["SHAP Value"].apply(
                lambda v: "Increases Risk 🔴" if v > 0 else "Reduces Risk 🔵"
            )
            shap_df["Color"] = shap_df["SHAP Value"].apply(
                lambda v: "#e74c3c" if v > 0 else "#3498db"
            )

            # SHAP bar chart
            fig_shap = go.Figure(go.Bar(
                x=shap_df["SHAP Value"],
                y=shap_df["Feature"],
                orientation="h",
                marker_color=shap_df["Color"].tolist(),
                text=[f"{v:+.4f}" for v in shap_df["SHAP Value"]],
                textposition="outside",
            ))
            fig_shap.update_layout(
                title="Feature Impact on Fraud Score (SHAP values)",
                xaxis_title="SHAP Value  →  impact on model output",
                yaxis={"autorange": "reversed"},
                height=max(300, len(feat_names) * 45),
                margin=dict(t=50, b=40, l=20, r=80),
                plot_bgcolor="white",
            )
            fig_shap.add_vline(x=0, line_width=1, line_color="black")
            st.plotly_chart(fig_shap, use_container_width=True)

            # ── PLAIN-ENGLISH EXPLANATION ────────────────────────────────────────
            st.subheader("📋 Plain-English Explanation")
            top_risk_feats = shap_df[shap_df["SHAP Value"] > 0].head(3)
            top_safe_feats = shap_df[shap_df["SHAP Value"] < 0].head(2)

            explain_html = '<div class="explain-box">'
            if not top_risk_feats.empty:
                explain_html += "<b>🔴 Risk drivers (pushing score UP):</b><ul>"
                for _, row in top_risk_feats.iterrows():
                    explain_html += (f"<li><b>{row['Feature']}</b> — "
                                     f"pushed the fraud score up by "
                                     f"<b>{abs(row['SHAP Value']):.4f}</b></li>")
                explain_html += "</ul>"
            if not top_safe_feats.empty:
                explain_html += "<b>🔵 Protective signals (pushing score DOWN):</b><ul>"
                for _, row in top_safe_feats.iterrows():
                    explain_html += (f"<li><b>{row['Feature']}</b> — "
                                     f"reduced the fraud score by "
                                     f"<b>{abs(row['SHAP Value']):.4f}</b></li>")
                explain_html += "</ul>"
            explain_html += "</div>"
            st.markdown(explain_html, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"SHAP explanation unavailable for this model type. "
                       f"(Use a tree-based model like RandomForest/XGBoost for SHAP.) "
                       f"Error: {e}")

        st.markdown("---")

        # ── TRANSACTION SUMMARY TABLE ────────────────────────────────────────────
        st.subheader("📄 Transaction Summary")
        summary = pd.DataFrame({
            "Field":  ["Type", "Amount", "Sender Before", "Sender After",
                       "Receiver Before", "Receiver After",
                       "Balance Drain Ratio", "Sender Mismatch", "Receiver Mismatch"],
            "Value":  [tx_type,
                       f"₹{amount:,.2f}",
                       f"₹{old_balance_orig:,.2f}",
                       f"₹{new_balance_orig:,.2f}",
                       f"₹{old_balance_dest:,.2f}",
                       f"₹{new_balance_dest:,.2f}",
                       f"{balance_drain_ratio:.2%}",
                       f"₹{sender_mismatch:,.2f}",
                       f"₹{receiver_mismatch:,.2f}"]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

else:
    # ── WELCOME SCREEN ───────────────────────────────────────────────────────────
    st.markdown("""
    ### 👈 Enter transaction details in the sidebar and click **Analyze Transaction**

    **What this app does:**
    - 🔢 **Risk Score (0–100)** — instantly shows how suspicious a transaction is
    - 🧠 **Explainable AI** — shows exactly *why* the model flagged it (SHAP values)
    - 📋 **Plain-English verdict** — no ML jargon, just clear action steps
    - 📊 **Feature impact chart** — which fields mattered most to the decision

    **Built with:** Python · Scikit-learn · Streamlit · SHAP · Plotly
    """)

    # Feature overview columns
    c1, c2, c3 = st.columns(3)
    c1.success("✅ **Low Risk (0–24)**\nTransaction proceeds normally")
    c2.warning("⚠️ **Medium Risk (25–49)**\nExtra authentication required")
    c3.error("🚨 **High / Critical (50–100)**\nFlag or block the transaction")
