"""Core logic: real data loading, feature engineering, models, scoring.
Enhanced for large-scale real transaction data with streaming support."""
import numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import os

# Feature list for model
FEATURES = ["log_amount", "amount_vs_mean", "log_time_gap", "hour_sin", "hour_cos",
            "balance_change_ratio", "dest_balance_change", "txn_velocity_1h", 
            "amount_velocity_1h", "is_transfer", "is_cash_out"]

def load_real_data(filepath, sample_frac=None, seed=42):
    """Load the real financial transactions dataset."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    
    if sample_frac and sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=seed)
        print(f"Sampled {len(df)} transactions ({sample_frac*100:.1f}%)")
    
    print(f"Loaded {len(df):,} transactions")
    print(f"Fraud rate: {df['isFraud'].mean():.4%}")
    print(f"Transaction types: {df['type'].value_counts().to_dict()}")
    
    return df

def build_features(df):
    """Engineer features from real transaction data."""
    print("Building features...")
    d = df.copy()
    
    # Sort by step (time) and customer
    d = d.sort_values(['nameOrig', 'step']).reset_index(drop=True)
    
    # Basic amount features
    d['log_amount'] = np.log1p(d['amount'])
    user_mean = d.groupby('nameOrig')['amount'].transform('mean')
    d['amount_vs_mean'] = np.log1p(d['amount'] / (user_mean + 1))
    
    # Time-based features
    g = d.groupby('nameOrig')
    d['time_gap'] = g['step'].diff().fillna(24).clip(lower=0.1)
    d['log_time_gap'] = np.log1p(d['time_gap'])
    
    # Hour encoding (treating step as hours in simulation)
    hour = (d['step'] % 24)
    d['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    d['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    
    # Balance change patterns
    d['balance_change_ratio'] = np.abs(d['newbalanceOrig'] - d['oldbalanceOrg']) / (d['oldbalanceOrg'] + 1)
    d['dest_balance_change'] = np.abs(d['newbalanceDest'] - d['oldbalanceDest']) / (d['oldbalanceDest'] + 1)
    
    # Velocity features: transaction count and amount in trailing window
    # Count transactions by same user in last hour (assuming step increments represent time units)
    cnt = np.ones(len(d))
    amt_sum = d['amount'].values.copy()
    
    for user, idx in d.groupby('nameOrig').indices.items():
        steps = d['step'].values[idx]
        amounts = d['amount'].values[idx]
        
        for i, (pos, curr_step) in enumerate(zip(idx, steps)):
            # Find transactions within 1 hour window
            window_mask = (steps <= curr_step) & (steps > curr_step - 1)
            cnt[pos] = np.sum(window_mask)
            amt_sum[pos] = np.sum(amounts[window_mask])
    
    d['txn_velocity_1h'] = np.log1p(cnt)
    d['amount_velocity_1h'] = np.log1p(amt_sum)
    
    # Transaction type indicators
    d['is_transfer'] = (d['type'] == 'TRANSFER').astype(int)
    d['is_cash_out'] = (d['type'] == 'CASH_OUT').astype(int)
    
    # Fill any NaN values
    for feat in FEATURES:
        d[feat] = d[feat].fillna(0).replace([np.inf, -np.inf], 0)
    
    print(f"Features built: {len(d)} records")
    return d

class Detector:
    """Isolation Forest + autoencoder (MLP trained to reproduce its input), rank-averaged."""
    def __init__(self, alert_rate=0.01, seed=42):
        self.alert_rate, self.seed = alert_rate, seed
        
    def fit(self, X):
        """Train on normal transactions."""
        print(f"Training detector on {len(X):,} samples...")
        self.sc = StandardScaler().fit(X)
        Z = self.sc.transform(X)
        
        # Isolation Forest
        print("  Training Isolation Forest...")
        self.iso = IsolationForest(
            n_estimators=300, 
            max_samples=min(5000, len(X)),
            random_state=self.seed,
            n_jobs=-1
        ).fit(Z)
        
        # Autoencoder (MLP)
        print("  Training Autoencoder...")
        self.ae = MLPRegressor(
            hidden_layer_sizes=(32, 8, 32), 
            activation="tanh",
            max_iter=500,
            learning_rate_init=3e-3,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=self.seed,
            verbose=False
        ).fit(Z, Z)
        
        # Compute reference distributions
        s = self._raw(Z)
        self.ref = {k: np.sort(v) for k, v in s.items()}
        self.threshold = np.quantile(self.score(X)["score"], 1 - self.alert_rate)
        
        print(f"  Training complete. Threshold: {self.threshold:.4f}")
        return self
        
    def _raw(self, Z):
        """Compute raw anomaly scores."""
        err = (Z - self.ae.predict(Z))**2
        return {
            "iso": -self.iso.score_samples(Z), 
            "ae": err.mean(1), 
            "ae_feat": err
        }
        
    def score(self, X):
        """Score transactions."""
        Z = self.sc.transform(X)
        r = self._raw(Z)
        
        # Percentile ranking vs training distribution
        pct = lambda k: np.searchsorted(self.ref[k], r[k]) / len(self.ref[k])
        
        return {
            "score": (pct("iso") + pct("ae")) / 2,
            "iso": r["iso"],
            "ae": r["ae"],
            "ae_feat": r["ae_feat"]
        }

def explain(row):
    """Human-readable reasons for flagging."""
    reasons = []
    
    if row.get('amount_vs_mean', 0) > 2.0:
        reasons.append(f"Amount {np.exp(row.amount_vs_mean):.1f}x user average")
    
    if row.get('balance_change_ratio', 0) > 5.0:
        reasons.append(f"Large balance change ratio: {row.balance_change_ratio:.1f}")
    
    if row.get('txn_velocity_1h', 0) > 2.5:
        reasons.append(f"High transaction velocity: {int(np.exp(row.txn_velocity_1h))} txns/hour")
    
    if row.get('amount_velocity_1h', 0) > 12:
        reasons.append(f"Unusual amount velocity")
    
    if row.get('type') in ['TRANSFER', 'CASH_OUT'] and row.get('amount', 0) > 200000:
        reasons.append(f"Large {row.type}: ₹{row.amount:,.0f}")
    
    if row.get('oldbalanceOrg', 0) > 0 and row.get('newbalanceOrig', 0) == 0:
        reasons.append("Account drained to zero")
    
    return reasons or ["Unusual pattern combination"]
