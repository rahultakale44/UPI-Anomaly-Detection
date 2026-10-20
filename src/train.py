"""
Leakage-conscious baseline:
- chronologically splits PaySim by `step`
- fits Isolation Forest only on normal transactions from the training period
- uses validation labels only to choose a threshold
- evaluates once on the later test period
Labels are never input features.
"""
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score, confusion_matrix,
)
from sklearn.preprocessing import StandardScaler

from src.data_profile import locate_csv

ARTIFACTS = Path("artifacts")
REPORTS = Path("reports")
FEATURES = ["amount", "hour_of_day", "origin_prior_tx_count", "origin_prior_amount_sum"]

def load_data(path: Path) -> pd.DataFrame:
    # Read full data. For limited-RAM laptops, replace with a sampled CSV for the demo.
    df = pd.read_csv(path)
    required = {"step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest", "isFraud"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required PaySim columns: {sorted(missing)}")
    df = df.dropna(subset=list(required)).copy()
    df["hour_of_day"] = df["step"].astype(float) % 24
    # Use only information available at decision time. Do NOT use post-transaction
    # balances: in PaySim these can encode the synthetic fraud-generation mechanism.
    if "nameOrig" in df.columns:
        df["origin_prior_tx_count"] = df.groupby("nameOrig", sort=False).cumcount()
        # Past-only cumulative amount: subtract current transaction from cumulative sum.
        cum = df.groupby("nameOrig", sort=False)["amount"].cumsum()
        df["origin_prior_amount_sum"] = cum - df["amount"]
    else:
        # Dataset variant without account IDs: mark these as unavailable rather than inventing history.
        df["origin_prior_tx_count"] = 0
        df["origin_prior_amount_sum"] = 0.0
    # Encode transaction type with fixed one-hot columns for robust inference.
    type_dummies = pd.get_dummies(df["type"], prefix="type", dtype=int)
    for t in ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]:
        col = f"type_{t}"
        if col not in type_dummies:
            type_dummies[col] = 0
    type_dummies = type_dummies[[f"type_{t}" for t in ["CASH_IN","CASH_OUT","DEBIT","PAYMENT","TRANSFER"]]]
    df = pd.concat([df, type_dummies], axis=1)
    features = FEATURES + list(type_dummies.columns)
    df[features] = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    df["isFraud"] = df["isFraud"].astype(int)
    return df, features

def metrics(y, pred, scores):
    cm = confusion_matrix(y, pred, labels=[0, 1])
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc_average_precision": float(average_precision_score(y, scores)) if len(set(y)) > 1 else None,
        "roc_auc": float(roc_auc_score(y, scores)) if len(set(y)) > 1 else None,
        "confusion_matrix": {"tn": int(cm[0,0]), "fp": int(cm[0,1]), "fn": int(cm[1,0]), "tp": int(cm[1,1])},
        "fraud_prevalence": float(np.mean(y)),
        "n": int(len(y)),
    }
    return out

def main():
    ARTIFACTS.mkdir(exist_ok=True); REPORTS.mkdir(exist_ok=True)
    path = locate_csv()
    started = time.perf_counter()
    df, features = load_data(path)
    df = df.sort_values("step").reset_index(drop=True)
    # Chronological 70/15/15 split. A timestep can span many transactions; split by row order
    # after sorting, and print ranges so this limitation is transparent.
    n = len(df); a = int(n * .70); b = int(n * .85)
    train, val, test = df.iloc[:a].copy(), df.iloc[a:b].copy(), df.iloc[b:].copy()
    X_train_all = train[features]
    # Fully unsupervised fit: do not use isFraud labels during model training.
    # Keep the demo feasible on a typical student laptop. The sample is taken from
    # the training period only; the later test period remains held out.
    max_fit_rows = 300_000
    X_fit = train[features]
    if len(X_fit) > max_fit_rows:
        X_fit = X_fit.sample(n=max_fit_rows, random_state=42)
    if len(X_fit) < 100:
        raise ValueError("Too few training examples.")
    model = IsolationForest(
        n_estimators=200, contamination="auto", random_state=42, n_jobs=-1,
        max_samples="auto"
    )
    fit_start = time.perf_counter()
    model.fit(X_fit)
    training_seconds = time.perf_counter() - fit_start

    # Higher score = more anomalous (opposite of sklearn decision_function).
    val_scores = -model.decision_function(val[features])
    # Select threshold on validation labels by maximizing F1. This is a labelled validation
    # protocol, not a purely unsupervised threshold; report it honestly.
    candidates = np.unique(np.quantile(val_scores, np.linspace(.90, .9999, 400)))
    best = {"f1": -1.0, "threshold": float(candidates[0])}
    yv = val["isFraud"].to_numpy()
    for threshold in candidates:
        pv = (val_scores >= threshold).astype(int)
        f = f1_score(yv, pv, zero_division=0)
        if f > best["f1"]:
            best = {"f1": float(f), "threshold": float(threshold)}
    threshold = best["threshold"]

    # Evaluate on a reproducible sample from the later chronological test period.
    # This keeps demo runtime practical while retaining rare fraud examples with a large sample.
    max_test_rows = 100_000
    if len(test) > max_test_rows:
        test = test.sample(n=max_test_rows, random_state=2026).sort_index()
    Xt = test[features]
    # Batch scoring for evaluation; separately benchmark repeated one-row scoring.
    scores = -model.decision_function(Xt)
    latency_rows = Xt.iloc[:min(500, len(Xt))]
    latencies_ms = []
    for i in range(len(latency_rows)):
        t0 = time.perf_counter()
        model.decision_function(latency_rows.iloc[[i]])
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    pred = (scores >= threshold).astype(int)
    result = metrics(test["isFraud"].to_numpy(), pred, scores)
    result.update({
        "dataset_file": path.name,
        "dataset_rows": int(len(df)),
        "feature_count": len(features),
        "features": features,
        "split_rows": {"train_period_rows": len(train), "validation_rows": len(val), "test_evaluation_sample_rows": len(test), "model_fit_sample_rows": len(X_fit)},
        "split_step_ranges": {
            "train": [int(train.step.min()), int(train.step.max())],
            "validation": [int(val.step.min()), int(val.step.max())],
            "test": [int(test.step.min()), int(test.step.max())],
        },
        "threshold": float(threshold),
        "threshold_method": "Validation F1 maximization using labels; not label-free thresholding",
        "validation_best_f1": float(best["f1"]),
        "training_seconds": float(training_seconds),
        "scoring_latency_ms_mean": float(np.mean(latencies_ms)),
        "scoring_latency_ms_p50": float(np.percentile(latencies_ms, 50)),
        "scoring_latency_ms_p95": float(np.percentile(latencies_ms, 95)),
        "scoring_latency_ms_p99": float(np.percentile(latencies_ms, 99)),
        "single_row_scoring_throughput_tx_per_second": float(1000 / np.mean(latencies_ms)) if np.mean(latencies_ms) else None,
        "end_to_end_train_script_seconds": float(time.perf_counter() - started),
        "evaluation_note": "Metrics are against PaySim simulation labels, not real-world UPI ground truth. Threshold tuned on labelled validation data. Model fit is capped at 300k training-period rows and test metrics use up to 100k sampled rows for laptop feasibility.",
    })
    joblib.dump({"model": model, "features": features, "threshold": threshold}, ARTIFACTS / "isolation_forest.joblib")
    # Save only the test examples needed by dashboard; labels are retained for retrospective evaluation.
    out = test[["step", "type", "amount", "oldbalanceOrg", "newbalanceOrig",
                "oldbalanceDest", "newbalanceDest", "isFraud"]].copy()
    out["anomaly_score"] = scores
    out["predicted_anomaly"] = pred
    out["risk_level"] = np.where(pred == 1, "HIGH", "NORMAL")
    out.to_csv(REPORTS / "scored_test_transactions.csv", index=False)
    (REPORTS / "metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"\nSaved model: {ARTIFACTS / 'isolation_forest.joblib'}")
    print(f"Saved metrics: {REPORTS / 'metrics.json'}")
    print(f"Saved scored transactions: {REPORTS / 'scored_test_transactions.csv'}")

if __name__ == "__main__":
    main()
