import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import shap
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FraudXPay", page_icon="🛡️", layout="wide")

# ── Load model ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("fraud_detection_pipeline.pkl")

model = load_model()

# ── Header ───────────────────────────────────────────────────────────────────────
st.title("🛡️ FraudXPay — Fraud Detection App")
st.markdown("Enter transaction details below and click **Predict** to get a fraud risk score with AI explanation.")
st.divider()

# ── Inputs (exactly same as your original) ────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    transaction_type = st.selectbox("Transaction Type", ["PAYMENT", "TRANSFER", "CASH_OUT", "DEPOSIT"])
    amount           = st.number_input("Amount",              min_value=0.0, value=10000.0)
    oldbalanceOrg    = st.number_input("Old Balance (Sender)", min_value=0.0, value=10000.0)
    newbalanceOrig   = st.number_input("New Balance (Sender)", min_value=0.0, value=9000.0)

with col2:
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    oldbalanceDest   = st.number_input("Old Balance (Receiver)", min_value=0.0, value=0.0)
    newbalanceDest   = st.number_input("New Balance (Receiver)", min_value=0.0, value=0.0)

st.divider()

# ── Predict button ───────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True):

    # Build input — exact same columns your pipeline expects
    input_data = pd.DataFrame([{
        "type":            transaction_type,
        "amount":          amount,
        "oldbalanceOrg":   oldbalanceOrg,
        "newbalanceOrig":  newbalanceOrig,
        "oldbalanceDest":  oldbalanceDest,
        "newbalanceDest":  newbalanceDest,
        "balanceDiffOrig": oldbalanceOrg - newbalanceOrig,
        "balanceDiffDest": newbalanceDest - oldbalanceDest,
    }])

    prediction = model.predict(input_data)[0]

    # ── Risk score from probability ───────────────────────────────────────────────
    try:
        fraud_prob = model.predict_proba(input_data)[0][1]
    except Exception:
        fraud_prob = float(prediction)

    risk_score = int(round(fraud_prob * 100))

    # ── Verdict helpers ───────────────────────────────────────────────────────────
    if risk_score < 25:
        verdict_text   = "✅ This transaction looks safe"
        verdict_color  = "#27ae60"
        recommendation = "Transaction can proceed normally."
        verdict_fn     = st.success
    elif risk_score < 50:
        verdict_text   = "⚠️ Possible anomaly — review before approving"
        verdict_color  = "#f39c12"
        recommendation = "Apply extra verification before proceeding."
        verdict_fn     = st.warning
    elif risk_score < 75:
        verdict_text   = "🔶 Suspicious — flag for manual review"
        verdict_color  = "#e67e22"
        recommendation = "Do not process. Send to compliance team."
        verdict_fn     = st.error
    else:
        verdict_text   = "🚨 Likely Fraud — block immediately"
        verdict_color  = "#e74c3c"
        recommendation = "Block transaction and notify the account holder."
        verdict_fn     = st.error

    # ── Gauge + Verdict ───────────────────────────────────────────────────────────
    left, right = st.columns([1, 1])

    with left:
        st.subheader("📊 Risk Score")
        gauge = go.Figure(go.Indicator(
            mode  = "gauge+number",
            value = risk_score,
            title = {"text": "Fraud Risk Score (0 = Safe · 100 = Fraud)"},
            number= {"suffix": " / 100"},
            gauge = {
                "axis" : {"range": [0, 100]},
                "bar"  : {"color": verdict_color, "thickness": 0.3},
                "steps": [
                    {"range": [0,  25], "color": "#d4edda"},
                    {"range": [25, 50], "color": "#fff3cd"},
                    {"range": [50, 75], "color": "#ffe0b2"},
                    {"range": [75,100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line"     : {"color": verdict_color, "width": 5},
                    "thickness": 0.85,
                    "value"    : risk_score
                }
            }
        ))
        gauge.update_layout(height=280, margin=dict(t=40, b=0, l=20, r=20))
        st.plotly_chart(gauge, use_container_width=True)

    with right:
        st.subheader("🏷️ Verdict")
        verdict_fn(verdict_text)
        st.info(f"**Recommendation:** {recommendation}")
        st.metric("Fraud Probability", f"{fraud_prob:.1%}")
        st.metric("Raw Prediction",    str(int(prediction)))

    st.divider()

    # ── Explainable AI ────────────────────────────────────────────────────────────
    st.subheader("🧠 Explainable AI — Why did the model decide this?")
    st.caption("🔴 Red = pushed score toward FRAUD  |  🔵 Blue = pushed score toward SAFE")

    try:
        from sklearn.pipeline import Pipeline

        # Extract last step (the classifier)
        step_names   = list(model.named_steps.keys())
        classifier   = model.named_steps[step_names[-1]]

        # Transform data through all steps except the classifier
        if len(step_names) > 1:
            pre_steps    = [(k, model.named_steps[k]) for k in step_names[:-1]]
            preprocessor = Pipeline(pre_steps)
            X_tf         = preprocessor.transform(input_data)
            try:
                feat_names = list(preprocessor.get_feature_names_out())
            except Exception:
                feat_names = [f"feature_{i}" for i in range(X_tf.shape[1])]
        else:
            X_tf       = input_data.values
            feat_names = list(input_data.columns)

        # SHAP — auto-detect model type and pick the right explainer
        from sklearn.linear_model import LogisticRegression, LinearRegression
        from sklearn.svm import SVC, SVR

        if isinstance(classifier, (LogisticRegression, LinearRegression)):
            # LinearExplainer works for linear models — needs a background dataset
            background = np.zeros((1, X_tf.shape[1]))
            explainer   = shap.LinearExplainer(classifier, background,
                                               feature_perturbation="interventional")
            shap_values = explainer.shap_values(X_tf)
            # LinearExplainer returns a single array (not a list)
            sv = shap_values[0] if shap_values.ndim == 2 else shap_values
        else:
            explainer   = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(X_tf)
            sv = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        n  = min(len(sv), len(feat_names))
        shap_df = pd.DataFrame({"Feature": feat_names[:n], "SHAP": sv[:n]})
        shap_df = shap_df.reindex(shap_df["SHAP"].abs().sort_values(ascending=False).index)

        colors = ["#e74c3c" if v > 0 else "#3498db" for v in shap_df["SHAP"]]

        fig = go.Figure(go.Bar(
            x            = shap_df["SHAP"],
            y            = shap_df["Feature"],
            orientation  = "h",
            marker_color = colors,
            text         = [f"{v:+.4f}" for v in shap_df["SHAP"]],
            textposition = "outside",
        ))
        fig.add_vline(x=0, line_width=1, line_color="black")
        fig.update_layout(
            xaxis_title  = "SHAP value  →  impact on fraud score",
            yaxis        = {"autorange": "reversed"},
            height       = max(300, n * 48),
            margin       = dict(t=20, b=30, l=10, r=80),
            plot_bgcolor = "white",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Plain-English explanation
        st.markdown("**📋 In plain English:**")
        risk_drivers = shap_df[shap_df["SHAP"] > 0].head(3)
        safe_signals = shap_df[shap_df["SHAP"] < 0].head(2)

        if not risk_drivers.empty:
            st.markdown("🔴 **These features increased the fraud score:**")
            for _, row in risk_drivers.iterrows():
                st.markdown(f"- **{row['Feature']}** raised fraud probability by `{abs(row['SHAP']):.4f}`")

        if not safe_signals.empty:
            st.markdown("🔵 **These features reduced the fraud score:**")
            for _, row in safe_signals.iterrows():
                st.markdown(f"- **{row['Feature']}** lowered fraud probability by `{abs(row['SHAP']):.4f}`")

    except Exception as e:
        # Fallback: use coef_ (Logistic Regression) or feature_importances_ (tree models)
        st.warning("Showing model coefficients as feature impact chart.")
        try:
            from sklearn.pipeline import Pipeline
            step_names  = list(model.named_steps.keys())
            clf         = model.named_steps[step_names[-1]]

            # Get transformed feature names for the fallback chart
            if len(step_names) > 1:
                pre_steps    = [(k, model.named_steps[k]) for k in step_names[:-1]]
                preprocessor = Pipeline(pre_steps)
                X_fb         = preprocessor.transform(input_data)
                try:
                    fb_names = list(preprocessor.get_feature_names_out())
                except Exception:
                    fb_names = [f"feature_{i}" for i in range(X_fb.shape[1])]
            else:
                fb_names = list(input_data.columns)

            # Pick coef_ for linear models, feature_importances_ for tree models
            if hasattr(clf, "coef_"):
                imp    = clf.coef_[0]          # Logistic Regression coefficients
                x_label = "Coefficient (positive = more fraud risk)"
                colors  = ["#e74c3c" if v > 0 else "#3498db" for v in imp]
            elif hasattr(clf, "feature_importances_"):
                imp    = clf.feature_importances_
                x_label = "Feature Importance"
                colors  = "#4a90d9"
            else:
                raise ValueError("Model has no coef_ or feature_importances_.")

            n   = min(len(imp), len(fb_names))
            fi_df = pd.DataFrame({
                "Feature": fb_names[:n],
                "Value":   imp[:n],
            }).reindex(pd.Series(imp[:n]).abs().sort_values(ascending=False).index)

            fig2 = go.Figure(go.Bar(
                x            = fi_df["Value"],
                y            = fi_df["Feature"],
                orientation  = "h",
                marker_color = colors if isinstance(colors, list) else [colors]*n,
                text         = [f"{v:+.4f}" for v in fi_df["Value"]],
                textposition = "outside",
            ))
            fig2.add_vline(x=0, line_width=1, line_color="black")
            fig2.update_layout(
                xaxis_title  = x_label,
                yaxis        = {"autorange": "reversed"},
                height       = max(300, n * 48),
                margin       = dict(t=20, b=30, l=10, r=80),
                plot_bgcolor = "white",
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Plain-English from coefficients
            st.markdown("**📋 In plain English** (based on model coefficients):")
            coef_df   = fi_df.copy()
            risk_rows = coef_df[coef_df["Value"] > 0].head(3)
            safe_rows = coef_df[coef_df["Value"] < 0].head(2)
            if not risk_rows.empty:
                st.markdown("🔴 **Features that increase fraud risk:**")
                for _, row in risk_rows.iterrows():
                    st.markdown(f"- **{row['Feature']}** — coefficient `{row['Value']:+.4f}`")
            if not safe_rows.empty:
                st.markdown("🔵 **Features that reduce fraud risk:**")
                for _, row in safe_rows.iterrows():
                    st.markdown(f"- **{row['Feature']}** — coefficient `{row['Value']:+.4f}`")

        except Exception as e2:
            st.info(f"Could not generate explanation. Error: {e2}")
