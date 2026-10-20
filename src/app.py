from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from src.genai_explainer import explain_with_gemini

st.set_page_config(page_title="UPIShield | Risk Monitor", page_icon="🛡️", layout="wide")
st.title("UPIShield")
st.caption("Unsupervised transaction anomaly detection • PaySim simulation • Not connected to live UPI rails")

metrics_path = Path("reports/metrics.json")
scored_path = Path("reports/scored_test_transactions.csv")
model_path = Path("artifacts/isolation_forest.joblib")

if not metrics_path.exists() or not scored_path.exists() or not model_path.exists():
    st.warning("Model artifacts are missing. Run `python -m src.train` first.")
    st.stop()

metrics = json.loads(metrics_path.read_text())
df = pd.read_csv(scored_path)
c1,c2,c3,c4 = st.columns(4)
c1.metric("Test transactions", f"{metrics['n']:,}")
c2.metric("Precision", f"{metrics['precision']:.3f}")
c3.metric("Recall", f"{metrics['recall']:.3f}")
c4.metric("F1 score", f"{metrics['f1']:.3f}")
c5,c6,c7,c8 = st.columns(4)
c5.metric("PR-AUC (Average Precision)", f"{metrics['pr_auc_average_precision']:.3f}" if metrics["pr_auc_average_precision"] is not None else "N/A")
c6.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}" if metrics["roc_auc"] is not None else "N/A")
c7.metric("Mean model latency", f"{metrics['scoring_latency_ms_mean']:.3f} ms")
c8.metric("p95 model latency", f"{metrics['scoring_latency_ms_p95']:.3f} ms")

st.info("Evaluation uses the dataset's simulation labels for retrospective scoring. These metrics are not proof of performance on real Indian UPI traffic. Latency shown is local model scoring latency, not network or GenAI response time.")
tab1, tab2, tab3 = st.tabs(["Transaction monitor", "Model evaluation", "Try a transaction"])

with tab1:
    anomalies = df[df.predicted_anomaly == 1].sort_values("anomaly_score", ascending=False)
    st.subheader("Highest-scored flagged transactions")
    st.dataframe(anomalies.head(100), use_container_width=True, hide_index=True)
    st.subheader("Anomaly score distribution")
    st.plotly_chart(px.histogram(df, x="anomaly_score", color="predicted_anomaly",
                                 nbins=60, title="Held-out test-set anomaly scores"), use_container_width=True)

with tab2:
    cm = metrics["confusion_matrix"]
    cm_df = pd.DataFrame([[cm["tn"], cm["fp"]],[cm["fn"], cm["tp"]]],
                         index=["Actual normal","Actual fraud"], columns=["Predicted normal","Predicted anomaly"])
    st.subheader("Confusion matrix")
    st.dataframe(cm_df, use_container_width=True)
    st.json({k:v for k,v in metrics.items() if k not in ["features"]})
    st.caption("Threshold method: " + metrics["threshold_method"])

with tab3:
    st.write("Score a manually entered transaction using the trained model. This is a model-only local measurement.")
    pack = joblib.load(model_path)
    with st.form("transaction_form"):
        amount = st.number_input("Amount", min_value=0.0, value=2500.0, step=100.0)
        tx_type = st.selectbox("Transaction type", ["PAYMENT","TRANSFER","CASH_OUT","CASH_IN","DEBIT"])
        old_org = st.number_input("Origin balance before", min_value=0.0, value=5000.0, step=100.0)
        new_org = st.number_input("Origin balance after", min_value=0.0, value=2500.0, step=100.0)
        old_dest = st.number_input("Destination balance before", min_value=0.0, value=1000.0, step=100.0)
        new_dest = st.number_input("Destination balance after", min_value=0.0, value=3500.0, step=100.0)
        step = st.number_input("Simulation step (hour index)", min_value=0, value=100, step=1)
        submit = st.form_submit_button("Score transaction")
    if submit:
        row = {
            "amount": amount,
            "hour_of_day": float(step % 24),
            "origin_prior_tx_count": 0,
            "origin_prior_amount_sum": 0.0,
        }
        for t in ["CASH_IN","CASH_OUT","DEBIT","PAYMENT","TRANSFER"]:
            row[f"type_{t}"] = int(tx_type == t)
        x = pd.DataFrame([{f: row.get(f, 0) for f in pack["features"]}])
        t0 = time.perf_counter()
        score = float(-pack["model"].decision_function(x)[0])
        latency = (time.perf_counter() - t0) * 1000
        flagged = score >= pack["threshold"]
        st.metric("Anomaly score", f"{score:.6f}")
        st.metric("Model scoring latency", f"{latency:.3f} ms")
        st.metric("Decision", "FLAG FOR REVIEW" if flagged else "NOT FLAGGED")
        evidence = []
        if amount > 100000: evidence.append("Very high transaction amount (fixed illustrative rule)")
        if old_org > 0 and new_org == 0: evidence.append("Origin balance becomes zero")
        if abs((old_org-new_org)-amount) > max(1.0, amount*0.05): evidence.append("Origin balance change differs materially from amount")
        if abs((new_dest-old_dest)-amount) > max(1.0, amount*0.05): evidence.append("Destination balance change differs materially from amount")
        st.write("Evidence checks (heuristics, not model explanations):")
        st.write(evidence if evidence else ["No listed heuristic triggered. The anomaly score may still be unusual relative to the fitted baseline."])
        st.subheader("Generative AI risk explanation")
        facts = {
            "decision": "FLAG FOR REVIEW" if flagged else "NOT FLAGGED",
            "anomaly_score": round(score, 6),
            "threshold": round(float(pack["threshold"]), 6),
            "transaction_type": tx_type,
            "amount": amount,
            "hour_index": int(step),
            "origin_balance_before": old_org,
            "origin_balance_after": new_org,
            "destination_balance_before": old_dest,
            "destination_balance_after": new_dest,
            "heuristic_evidence": evidence,
            "important_limit": "This is simulated data and an anomaly is not proof of fraud."
        }
        with st.spinner("Generating grounded explanation with Gemini..."):
            try:
                explanation, genai_ms = explain_with_gemini(facts)
                st.write(explanation)
                st.caption(f"GenAI API latency: {genai_ms:.0f} ms (separate from model latency).")
            except Exception as exc:
                st.error(f"GenAI explanation unavailable: {exc}")
                st.info("Check that GEMINI_API_KEY is set in your .env file and that your API quota is available.")
        st.caption("A flagged transaction is an anomaly candidate, not proof of fraud. The GenAI model receives the calculated evidence and is instructed not to invent missing information.")
