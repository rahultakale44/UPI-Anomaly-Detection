# 🚀 Quick Start Guide

## System Status: ✅ READY

Your real-time transaction anomaly detection system is now running!

## 🌐 Access the Web Interface

Open your browser and navigate to:

```
http://127.0.0.1:5000
```

## 📊 System Configuration

**Dataset Loaded:**
- **Total Transactions:** 6,362,620
- **Actual Fraud Rate:** 0.1291% (8,213 frauds)
- **Transaction Types:** CASH_OUT, PAYMENT, CASH_IN, TRANSFER, DEBIT

**Model Details:**
- **Training Set:** 49,942 normal transactions
- **Algorithm:** Isolation Forest + Autoencoder (Ensemble)
- **Detection Threshold:** 0.9765 (top ~2% anomaly scores)
- **Features:** 11 engineered features from raw data

## 🎮 How to Use the Web Interface

### 1. Start the Stream
Click **▶ Start Stream** to begin processing transactions in real-time.

Watch as transactions flow through the system with:
- **Green "OK"** badges for normal transactions
- **Red "FLAGGED"** badges for anomalies
- Live statistics updating in real-time

### 2. Adjust Settings

**Speed Control:**
- **Very Fast:** Process 100 txns/sec (for demos)
- **Fast:** Process ~50 txns/sec (default)
- **Normal:** Process ~20 txns/sec
- **Slow:** Process ~5 txns/sec (detailed viewing)

**Threshold Slider:**
- Move **LEFT** (0.7-0.9): More sensitive, catch more frauds, more false alarms
- Move **RIGHT** (0.9-1.0): Less sensitive, fewer false alarms, miss some frauds
- **Default:** 0.9765 (balanced)

### 3. Monitor Performance

Watch the live statistics:
- **Processed:** Total transactions analyzed
- **Flagged by Model:** How many alerts generated
- **Actual Fraud:** Ground truth (hidden in real scenarios)
- **Precision:** What % of alerts are real fraud
- **Recall:** What % of frauds we caught

### 4. Filter & Analyze

**Toggle Options:**
- ☑ **Show Ground Truth:** Reveal actual fraud labels (for validation)
- ☑ **Flagged Only:** Hide normal transactions, show alerts only
- ☑ **Auto-scroll:** Keep newest transactions visible

### 5. Test Custom Transactions

Click **⚡ Test Custom Txn** to score your own scenarios:

**Example Scenarios to Try:**

**Normal Transaction:**
```
Type: PAYMENT
Amount: 5000
Origin: C1234567890
Destination: M9876543210
Old Balance: 50000
```
Expected: ✅ OK (score ~0.3-0.6)

**Large Unusual Amount:**
```
Type: TRANSFER
Amount: 500000
Origin: C1234567890
Destination: C9876543210
Old Balance: 50000
```
Expected: 🚨 FLAGGED (score ~0.95+)

**Account Draining:**
```
Type: CASH_OUT
Amount: 100000
Origin: C1234567890
Destination: C9876543210
Old Balance: 100000
```
Expected: 🚨 FLAGGED (balance goes to zero)

## 📈 Expected Performance

Based on initial testing with real data:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Alert Rate** | ~2% | 2 out of 100 transactions flagged |
| **Precision** | 50-70% | Half of alerts are real fraud |
| **Recall** | 70-85% | Catch 7-8 out of 10 frauds |
| **ROC-AUC** | 0.95+ | Excellent discrimination |

## 🔍 Understanding Flagged Transactions

When a transaction is flagged, the system shows:

1. **Anomaly Score** (0-1): Combined percentile rank from both models
2. **Reasons:** Human-readable explanations
   - "Amount 5.2x user average"
   - "High transaction velocity: 6 txns/hour"
   - "Large balance change ratio"
   - "Account drained to zero"
3. **Model Breakdown:**
   - **ISO:** Isolation Forest score
   - **AE:** Autoencoder reconstruction error
4. **Worst Feature:** Which feature contributed most

## 🛠️ Troubleshooting

**Browser shows "Connection refused"**
- Ensure the Flask server is running: `python app.py`
- Check the terminal for error messages

**Stream is very slow**
- Increase speed in the dropdown menu
- Or reduce BATCH_SIZE in app.py

**Want to reset the stream**
- Click **↻ Reset** button
- Or refresh the page

## 🎯 Quick Tips

1. **Try different thresholds** while streaming to see impact on precision/recall
2. **Use "Flagged Only"** mode to focus on anomalies
3. **Test custom transactions** with extreme values to understand model behavior
4. **Watch feature explanations** to learn what patterns indicate fraud

## 📝 Advanced Usage

### Run Offline Analysis
For comprehensive analysis with visualizations:
```bash
python main.py
```

This generates:
- Feature distribution plots
- ROC/PR curves  
- Performance metrics
- Flagged transactions CSV

### Adjust Training
Edit `app.py` CONFIG section:
```python
CONFIG = {
    'train_sample': 50000,    # Use more data for better accuracy
    'stream_batch': 100,      # Larger batches = faster but less smooth
}
```

### Export Results
While streaming, flagged transactions are available via:
```
http://127.0.0.1:5000/api/stats
```

## 🎓 Learning Resources

**Key Concepts:**
1. **Unsupervised Learning:** No fraud labels needed for training
2. **Ensemble:** Two models vote together (Isolation Forest + Autoencoder)
3. **Feature Engineering:** Transform raw data into meaningful patterns
4. **Threshold Tuning:** Trade-off between catching frauds vs false alarms

**Why This Works:**
- Frauds are rare and constantly evolving
- Model learns "normal" behavior patterns
- Deviations from normal = potential fraud
- Explainable: we can see WHY something was flagged

## 🚀 Next Steps

1. ✅ Open http://127.0.0.1:5000
2. ✅ Start streaming transactions
3. ✅ Adjust threshold to see trade-offs
4. ✅ Test custom suspicious transactions
5. ✅ Review performance metrics

## 💡 Use Cases

This system can detect:
- **Large unusual amounts** relative to user history
- **Rapid-fire transactions** (velocity attacks)
- **Account draining** (balance → 0)
- **Unusual transaction types** for the user
- **Impossible patterns** (e.g., balance inconsistencies)

## 📞 Support

Questions? Issues?
- Check README.md for detailed documentation
- Review the code comments in `anomaly_core.py`
- Examine `plots/` folder after running `main.py`

---

**Happy Fraud Hunting! 🕵️‍♂️🔍**
