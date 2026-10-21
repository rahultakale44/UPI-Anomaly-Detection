# %% Imports & setup
import os, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix
from anomaly_core import *

os.makedirs("plots", exist_ok=True)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

def show(name):
    """Save chart to ./plots."""
    plt.tight_layout()
    plt.savefig(f"plots/{name}.png", dpi=120)
    plt.close()

def title(t):
    print("\n" + "=" * 80 + f"\n{t}\n" + "=" * 80)

# %% STEP 1 - Load Real Dataset
title("STEP 1: Load Real Financial Transaction Dataset")

DATA_PATH = '../data/PS_20174392719_1491204439457_log.csv'

# Load dataset (use full dataset or sample for faster experimentation)
USE_SAMPLE = True  # Set to False to use full 6M+ transactions
SAMPLE_SIZE = 100000 if USE_SAMPLE else None

if SAMPLE_SIZE:
    print(f"Loading sample of {SAMPLE_SIZE:,} transactions for analysis...")
    # Read file in chunks to sample efficiently
    chunks = []
    chunk_size = 100000
    total_read = 0
    
    for chunk in pd.read_csv(DATA_PATH, chunksize=chunk_size):
        chunks.append(chunk)
        total_read += len(chunk)
        if total_read >= SAMPLE_SIZE:
            break
    
    raw = pd.concat(chunks, ignore_index=True).iloc[:SAMPLE_SIZE]
else:
    print("Loading full dataset...")
    raw = pd.read_csv(DATA_PATH)

print(f"\nDataset loaded: {len(raw):,} transactions")
print(f"Fraud rate: {raw['isFraud'].mean():.4%}")
print(f"Flagged fraud rate: {raw['isFlaggedFraud'].mean():.4%}")
print(f"\nTransaction types:\n{raw['type'].value_counts()}")
print(f"\nAmount statistics:")
print(raw['amount'].describe())

# %% STEP 2 - Feature Engineering
title("STEP 2: Feature Engineering on Real Data")

df = build_features(raw)
print(f"\nFeatures engineered for {len(df):,} transactions")
print(f"\nFeature statistics:")
print(df[FEATURES].describe().T[["mean", "std", "min", "max"]].round(3))

# Visualize feature distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

for idx, feature in enumerate(FEATURES[:6]):
    ax = axes[idx]
    normal = df[df['isFraud'] == 0][feature].dropna()
    fraud = df[df['isFraud'] == 1][feature].dropna()
    
    # Sample for visualization if too large
    if len(normal) > 10000:
        normal = normal.sample(10000, random_state=42)
    if len(fraud) > 1000:
        fraud = fraud.sample(min(1000, len(fraud)), random_state=42)
    
    ax.hist(normal, bins=50, alpha=0.6, density=True, label='Normal', color='green')
    ax.hist(fraud, bins=50, alpha=0.6, density=True, label='Fraud', color='red')
    ax.set_title(feature, fontsize=10)
    ax.legend()
    ax.set_ylabel('Density')

plt.suptitle('Feature Distributions: Normal vs Fraud', fontsize=14, fontweight='bold')
show("01_feature_distributions")

# %% STEP 3 - Split data and train model
title("STEP 3: Train Unsupervised Anomaly Detector")

# Use first 70% for training (normal transactions only)
train_size = int(0.7 * len(df))
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Train only on normal transactions
train_normal = train_df[train_df['isFraud'] == 0].copy()
print(f"\nTraining set: {len(train_normal):,} normal transactions")
print(f"Test set: {len(test_df):,} transactions ({test_df['isFraud'].sum():,} frauds)")

X_train = train_normal[FEATURES].values
detector = Detector(alert_rate=0.02, seed=42).fit(X_train)

# %% STEP 4 - Score test set
title("STEP 4: Score Test Set")

X_test = test_df[FEATURES].values
scores = detector.score(X_test)

test_df = test_df.copy()
test_df['anomaly_score'] = scores['score']
test_df['iso_score'] = scores['iso']
test_df['ae_score'] = scores['ae']

print(f"\nScoring complete!")
print(f"Score range: [{test_df['anomaly_score'].min():.3f}, {test_df['anomaly_score'].max():.3f}]")
print(f"Mean score: {test_df['anomaly_score'].mean():.3f}")
print(f"Std score: {test_df['anomaly_score'].std():.3f}")

# Visualize score distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].hist(test_df[test_df['isFraud']==0]['anomaly_score'], bins=60, alpha=0.7, label='Normal', density=True)
axes[0].hist(test_df[test_df['isFraud']==1]['anomaly_score'], bins=60, alpha=0.7, label='Fraud', density=True)
axes[0].axvline(detector.threshold, color='red', linestyle='--', label='Threshold')
axes[0].set_title('Combined Anomaly Score')
axes[0].set_xlabel('Score')
axes[0].legend()

axes[1].hist(np.log10(test_df['iso_score'] + 1e-10), bins=60, alpha=0.7)
axes[1].set_title('Isolation Forest Score (log scale)')
axes[1].set_xlabel('log10(ISO Score)')

axes[2].hist(np.log10(test_df['ae_score'] + 1e-10), bins=60, alpha=0.7)
axes[2].set_title('Autoencoder Reconstruction Error (log scale)')
axes[2].set_xlabel('log10(AE Error)')

plt.suptitle('Anomaly Score Distributions', fontsize=14, fontweight='bold')
show("02_score_distributions")

# %% STEP 5 - Threshold Analysis
title("STEP 5: Threshold Selection & Alert Budget")

thresholds = np.percentile(test_df['anomaly_score'], [90, 95, 97, 98, 99, 99.5, 99.9])
print("\nThreshold analysis (percentiles):")
print(f"{'Percentile':<12} {'Threshold':<12} {'Alerts':<10} {'Alert Rate':<12} {'Precision':<12} {'Recall'}")
print("-" * 80)

for pct, thr in zip([90, 95, 97, 98, 99, 99.5, 99.9], thresholds):
    flags = test_df['anomaly_score'] >= thr
    alert_rate = flags.sum() / len(test_df)
    
    if flags.sum() > 0:
        precision = (test_df[flags]['isFraud'] == 1).sum() / flags.sum()
        recall = (test_df[flags]['isFraud'] == 1).sum() / test_df['isFraud'].sum()
    else:
        precision = recall = 0
    
    print(f"{pct:<12.1f} {thr:<12.4f} {flags.sum():<10,} {alert_rate:<12.2%} {precision:<12.2%} {recall:.2%}")

print(f"\nChosen threshold: {detector.threshold:.4f}")

# Elbow plot
sorted_scores = np.sort(test_df['anomaly_score'].values)[::-1]
plt.figure(figsize=(10, 4))
plt.plot(sorted_scores[:2000])
plt.axhline(detector.threshold, color='r', linestyle='--', label=f'Threshold = {detector.threshold:.3f}')
plt.xlabel('Transaction Rank')
plt.ylabel('Anomaly Score')
plt.title('Score Distribution (Top 2000 transactions)')
plt.legend()
plt.grid(alpha=0.3)
show("03_threshold_elbow")

# %% STEP 6 - Performance Evaluation
title("STEP 6: Model Performance Evaluation")

# Apply threshold
test_df['flagged'] = test_df['anomaly_score'] >= detector.threshold

# Confusion matrix
cm = confusion_matrix(test_df['isFraud'], test_df['flagged'])
tn, fp, fn, tp = cm.ravel()

print("\nConfusion Matrix:")
print(f"{'':>15} {'Predicted Normal':>18} {'Predicted Fraud':>18}")
print(f"{'Actual Normal':>15} {tn:>18,} {fp:>18,}")
print(f"{'Actual Fraud':>15} {fn:>18,} {tp:>18,}")

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\nPerformance Metrics:")
print(f"Precision: {precision:.2%} ({tp}/{tp+fp})")
print(f"Recall: {recall:.2%} ({tp}/{tp+fn})")
print(f"F1-Score: {f1:.3f}")
print(f"False Positive Rate: {fp/(fp+tn):.2%}")

# ROC and PR curves
roc_auc = roc_auc_score(test_df['isFraud'], test_df['anomaly_score'])
pr_auc = average_precision_score(test_df['isFraud'], test_df['anomaly_score'])

print(f"\nAUC Metrics:")
print(f"ROC-AUC: {roc_auc:.4f}")
print(f"PR-AUC: {pr_auc:.4f}")

# Plot PR curve
precision_vals, recall_vals, _ = precision_recall_curve(test_df['isFraud'], test_df['anomaly_score'])

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(recall_vals, precision_vals, linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title(f'Precision-Recall Curve (AUC = {pr_auc:.4f})')
plt.grid(alpha=0.3)

# Feature importance from autoencoder errors
plt.subplot(1, 2, 2)
ae_errors = pd.DataFrame(scores['ae_feat'], columns=FEATURES)
flagged_mask = test_df['flagged'].values
feature_importance = ae_errors[flagged_mask].mean().sort_values(ascending=True)
feature_importance.plot.barh()
plt.xlabel('Mean Reconstruction Error')
plt.title('Feature Importance (Flagged Transactions)')
plt.tight_layout()
show("04_performance_metrics")

# %% STEP 7 - Analyze Flagged Transactions
title("STEP 7: Analyze Top Anomalies")

flagged_df = test_df[test_df['flagged']].sort_values('anomaly_score', ascending=False).head(20)

print("\nTop 20 Flagged Transactions:")
print(flagged_df[['step', 'type', 'amount', 'anomaly_score', 'isFraud']].to_string(index=False))

print("\n\nFraud Detection by Transaction Type:")
for txn_type in test_df['type'].unique():
    type_df = test_df[test_df['type'] == txn_type]
    type_frauds = type_df['isFraud'].sum()
    type_flagged = type_df['flagged'].sum()
    type_tp = ((type_df['isFraud'] == 1) & (type_df['flagged'])).sum()
    
    if type_frauds > 0:
        recall = type_tp / type_frauds
    else:
        recall = 0
    
    if type_flagged > 0:
        precision = type_tp / type_flagged
    else:
        precision = 0
    
    print(f"{txn_type:>12}: {type_frauds:>6,} frauds | Flagged: {type_flagged:>6,} | Precision: {precision:>6.1%} | Recall: {recall:>6.1%}")

# %% STEP 8 - Save Results
title("STEP 8: Save Results")

# Save flagged transactions
output_df = test_df[test_df['flagged']].sort_values('anomaly_score', ascending=False)
output_df.to_csv('flagged_transactions.csv', index=False)
print(f"\nSaved {len(output_df):,} flagged transactions to flagged_transactions.csv")

# Save model summary
with open('plots/model_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("ANOMALY DETECTION MODEL SUMMARY\n")
    f.write("="*80 + "\n\n")
    f.write(f"Dataset: {len(df):,} transactions\n")
    f.write(f"Training: {len(train_normal):,} normal transactions\n")
    f.write(f"Testing: {len(test_df):,} transactions ({test_df['isFraud'].sum():,} frauds)\n")
    f.write(f"Fraud rate: {test_df['isFraud'].mean():.4%}\n\n")
    f.write(f"Threshold: {detector.threshold:.4f}\n")
    f.write(f"Alert rate: {test_df['flagged'].mean():.2%}\n\n")
    f.write(f"Performance:\n")
    f.write(f"  Precision: {precision:.2%}\n")
    f.write(f"  Recall: {recall:.2%}\n")
    f.write(f"  F1-Score: {f1:.3f}\n")
    f.write(f"  ROC-AUC: {roc_auc:.4f}\n")
    f.write(f"  PR-AUC: {pr_auc:.4f}\n")

print("Saved model summary to plots/model_summary.txt")
print("\n" + "="*80)
print("ANALYSIS COMPLETE! Check the 'plots' folder for visualizations.")
print("="*80)
