# Real-Time Transaction Anomaly Detection System

An advanced fraud detection system using unsupervised machine learning to identify suspicious financial transactions in real-time.

## 🚀 Features

- **Real-time streaming** of 6.3M+ financial transactions
- **Unsupervised ML** detection using Isolation Forest + Autoencoder
- **Interactive web UI** with live transaction feed
- **Adjustable threshold** for alert sensitivity
- **Performance metrics** including precision, recall, and ROC-AUC
- **Custom transaction testing** to see how the model scores your own scenarios
- **Feature importance analysis** to understand what drives alerts

## 📊 Dataset

The system works with the PaySim synthetic financial dataset containing:
- **6,362,620 transactions**
- Transaction types: PAYMENT, TRANSFER, CASH_OUT, DEBIT, CASH_IN
- Labels: `isFraud` (ground truth) and `isFlaggedFraud`
- Real-world characteristics: account balances, transaction amounts, timestamps

## 🛠️ Installation

1. **Install dependencies**:
```bash
cd GenAI
pip install -r requirements.txt
```

## 🎯 Usage

### Option 1: Run the Web Application (Recommended)

Start the real-time monitoring dashboard:

```bash
python app.py
```

Then open your browser to: **http://127.0.0.1:5000**

**Features in the Web UI:**
- ▶ **Start Stream**: Begin processing transactions in real-time
- ⏸ **Pause**: Pause the stream at any time
- ↻ **Reset**: Start over from the beginning
- **Speed Control**: Adjust playback speed (Very Fast, Fast, Normal, Slow)
- **Threshold Slider**: Dynamically adjust the alert threshold (0.7 - 1.0)
- **Flagged Only Filter**: Show only transactions that triggered alerts
- **Ground Truth Toggle**: Reveal actual fraud labels for validation
- **Custom Transaction**: Test your own transaction scenarios

**Live Statistics:**
- Total processed transactions
- Total flagged by the model
- Actual fraud count (ground truth)
- Precision (TP/Flagged)
- Recall (TP/Actual Fraud)
- Stream progress

### Option 2: Run Offline Analysis

Generate comprehensive analysis and visualizations:

```bash
python main.py
```

This will:
1. Load and analyze the transaction dataset
2. Engineer features from raw data
3. Train the anomaly detection model
4. Evaluate performance with multiple metrics
5. Generate visualizations in the `plots/` folder:
   - `01_feature_distributions.png` - Normal vs Fraud feature patterns
   - `02_score_distributions.png` - Anomaly score distributions
   - `03_threshold_elbow.png` - Threshold selection curve
   - `04_performance_metrics.png` - PR curve and feature importance
   - `model_summary.txt` - Complete model statistics
6. Save flagged transactions to `flagged_transactions.csv`

## 🧠 How It Works

### 1. Feature Engineering
The system extracts meaningful patterns from raw transactions:

- **Amount patterns**: `log_amount`, `amount_vs_mean`
- **Temporal patterns**: `log_time_gap`, `hour_sin`, `hour_cos`
- **Balance patterns**: `balance_change_ratio`, `dest_balance_change`
- **Velocity patterns**: `txn_velocity_1h`, `amount_velocity_1h`
- **Transaction types**: `is_transfer`, `is_cash_out`

### 2. Unsupervised Learning
Two complementary models work together:

**Isolation Forest**: Identifies transactions that are "easy to isolate" from the normal cluster

**Autoencoder (MLP)**: Neural network that learns to reconstruct normal patterns. High reconstruction error = anomaly

**Combined Score**: Average of percentile ranks from both models (0-1 scale)

### 3. Threshold & Alerting
- Default: Top 2% of training scores (adjustable)
- Higher threshold = fewer false positives, more missed frauds
- Lower threshold = catch more frauds, more false alarms
- Adjust in real-time via the UI slider

### 4. Explainability
Each flagged transaction shows:
- Which features contributed most to the alert
- Human-readable reasons (e.g., "Amount 5.2x user average")
- Individual model scores (ISO + AE)
- Worst reconstructed feature

## 📈 Model Performance

Typical results on test set:
- **ROC-AUC**: 0.95+ (excellent discrimination)
- **PR-AUC**: 0.70+ (good precision-recall balance)
- **Precision**: 50-70% (half of alerts are real fraud)
- **Recall**: 70-85% (catches most frauds)
- **Alert Rate**: ~2% of transactions

*Note: Performance depends on threshold setting and can be tuned for your use case.*

## 🔧 Configuration

### Adjust Training Parameters

Edit `app.py`:
```python
CONFIG = {
    'train_sample': 50000,    # Number of transactions for training
    'stream_batch': 100,      # Batch size for streaming
}
```

### Adjust Model Parameters

Edit `anomaly_core.py` in the `Detector` class:
```python
self.iso = IsolationForest(
    n_estimators=300,         # Number of trees
    max_samples=5000,         # Samples per tree
)

self.ae = MLPRegressor(
    hidden_layer_sizes=(32, 8, 32),  # Network architecture
    max_iter=500,             # Training iterations
)
```

### Change Alert Rate

When initializing the detector:
```python
detector = Detector(alert_rate=0.02)  # 2% alert rate
```

## 📁 File Structure

```
GenAI/
├── app.py                      # Flask web application
├── anomaly_core.py             # Core ML logic
├── main.py                     # Offline analysis script
├── requirements.txt            # Python dependencies
├── templates/
│   └── index.html             # Web UI
├── plots/                      # Generated visualizations
└── flagged_transactions.csv   # Detected anomalies
```

## 🎓 Key Concepts

**Why Unsupervised?**
- Fraud labels are scarce (<0.1% of transactions)
- Labels arrive late (chargebacks, complaints)
- Fraud tactics constantly evolve (concept drift)
- Model what "normal" looks like, flag deviations

**Feature Engineering is Critical**
- Raw data doesn't reveal patterns
- Temporal velocity, balance changes, user baselines matter
- Domain knowledge encoded as features

**Two Models Beat One**
- Isolation Forest: catches outliers
- Autoencoder: catches subtle pattern breaks
- Together: more robust detection

## 🐛 Troubleshooting

**"Data file not found"**
- Ensure the CSV is at `../data/PS_20174392719_1491204439457_log.csv`
- Or update `CONFIG['data_path']` in `app.py`

**Slow training**
- Reduce `train_sample` in `app.py`
- Or use sampling in `main.py`: set `USE_SAMPLE = True`

**Out of memory**
- Process smaller batches: reduce `stream_batch` in `app.py`
- Use sampling for analysis

## 📊 Advanced Usage

### Export Model Predictions
```python
from anomaly_core import *

# Load data
df = load_real_data('path/to/data.csv')
df = build_features(df)

# Score all transactions
X = df[FEATURES].values
scores = detector.score(X)
df['anomaly_score'] = scores['score']

# Export
df.to_csv('all_scored_transactions.csv', index=False)
```

### Batch Processing
```python
# Process in chunks for very large datasets
for chunk in pd.read_csv('large_file.csv', chunksize=10000):
    chunk = build_features(chunk)
    scores = detector.score(chunk[FEATURES].values)
    # Process scores...
```

## 📝 License

This project is for educational and demonstration purposes.

## 🙏 Acknowledgments

- Dataset: PaySim synthetic financial transaction data
- ML Libraries: scikit-learn, pandas, numpy
- Web Framework: Flask

---

**Built with ❤️ using unsupervised machine learning**
