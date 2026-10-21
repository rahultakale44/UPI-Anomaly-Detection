# 🚀 Quick Reference Card

## Instant Access

### 🌐 Web Application
```
http://127.0.0.1:5000
```

### ⚡ Quick Commands
```bash
# Start the application
cd GenAI
python app.py

# Run offline analysis
python main.py

# Run system tests
python test_system.py

# Install dependencies
pip install -r requirements.txt
```

---

## 🎮 Web UI Controls

### Main Buttons
| Button | Action |
|--------|--------|
| **▶ Start Stream** | Begin processing transactions |
| **⏸ Pause** | Pause the stream |
| **↻ Reset** | Start over from beginning |
| **⚡ Test Custom Txn** | Score your own transaction |

### Settings
| Control | Purpose | Range |
|---------|---------|-------|
| **Speed** | Playback speed | Very Fast → Slow |
| **Threshold** | Alert sensitivity | 0.7 - 1.0 |
| **Flagged Only** | Filter view | On/Off |
| **Ground Truth** | Show labels | On/Off |
| **Auto-scroll** | Follow feed | On/Off |

---

## 📊 Understanding Scores

### Anomaly Score (0-1)
- **0.0 - 0.7:** Normal ✅
- **0.7 - 0.9:** Suspicious ⚠️
- **0.9 - 0.98:** Likely fraud 🚨
- **0.98 - 1.0:** Very likely fraud 🔴

### Default Threshold: **0.977**
- Above threshold → **FLAGGED**
- Below threshold → **OK**

### Model Components
- **ISO:** Isolation Forest score
- **AE:** Autoencoder reconstruction error
- **Combined:** Average percentile rank

---

## 🎯 Common Tasks

### Test a Normal Transaction
```javascript
Type: PAYMENT
Amount: 5000
Origin: C1234567890
Destination: M9876543210
Old Balance: 50000
Expected: ✅ OK (score ~0.3-0.6)
```

### Test a Suspicious Transaction
```javascript
Type: TRANSFER
Amount: 500000
Origin: C1234567890
Destination: C9876543210
Old Balance: 50000
Expected: 🚨 FLAGGED (score ~0.95+)
```

### Adjust for More Alerts
- Move threshold slider **LEFT** (e.g., to 0.90)
- Watch alert rate increase
- Precision will decrease, recall will increase

### Adjust for Fewer False Alarms
- Move threshold slider **RIGHT** (e.g., to 0.99)
- Watch alert rate decrease
- Precision will increase, recall will decrease

---

## 📈 Key Metrics Explained

### Precision
**What % of alerts are real fraud?**
- Formula: True Positives / (True Positives + False Positives)
- Target: >50%
- Higher is better

### Recall
**What % of frauds did we catch?**
- Formula: True Positives / (True Positives + False Negatives)
- Target: >60%
- Higher is better

### Alert Rate
**What % of transactions are flagged?**
- Typical: 2-5%
- Too low: Missing frauds
- Too high: Alert fatigue

---

## 🔍 Fraud Patterns Detected

### 1. Large Unusual Amounts
```
Pattern: Transaction >> user's average
Feature: amount_vs_mean > 2.0
Example: User avg ₹5k, transaction ₹50k
```

### 2. Rapid-Fire Transactions
```
Pattern: Multiple txns in short time
Feature: txn_velocity_1h > 4
Example: 6 transactions in 1 hour
```

### 3. Account Draining
```
Pattern: Balance → zero
Feature: newbalanceOrig = 0
Example: ₹100k → ₹0 in one transaction
```

### 4. Large Transfers/Cash-outs
```
Pattern: High-risk type + large amount
Features: is_transfer + amount > 200k
Example: ₹500k TRANSFER
```

---

## 🛠️ Configuration Quick Edit

### Change Training Size
**File:** `app.py`
```python
CONFIG = {
    'train_sample': 50000,    # ← Edit here
}
```

### Change Alert Rate
**File:** `app.py`
```python
DETECTOR = Detector(alert_rate=0.02)  # ← 2% alert rate
```

### Change Stream Speed
**UI:** Speed dropdown
- Very Fast: ~100 txns/sec
- Fast: ~50 txns/sec (default)
- Normal: ~20 txns/sec
- Slow: ~5 txns/sec

---

## 📁 File Quick Reference

### Essential Files
| File | Purpose |
|------|---------|
| `app.py` | Web server |
| `anomaly_core.py` | ML engine |
| `templates/index.html` | UI |
| `requirements.txt` | Dependencies |

### Documentation
| File | Content |
|------|---------|
| `STATUS.md` | Current status |
| `QUICK_START.md` | Tutorial |
| `README.md` | Full docs |
| `PROJECT_SUMMARY.md` | Technical details |

### Outputs
| File | Content |
|------|---------|
| `plots/*.png` | Visualizations |
| `flagged_transactions.csv` | Detected frauds |

---

## 🐛 Quick Troubleshooting

### Can't access http://127.0.0.1:5000
```bash
# Check if Flask is running
# Look for: "Running on http://127.0.0.1:5000"

# Try alternative URL
http://localhost:5000
```

### Server won't start
```bash
# Check data file exists
ls ../data/PS_20174392719_1491204439457_log.csv

# Check dependencies
pip install -r requirements.txt
```

### Out of memory
```python
# In app.py, reduce training size:
'train_sample': 10000,  # Instead of 50000
```

### Too slow
```python
# In app.py, reduce batch size:
'stream_batch': 50,  # Instead of 100
```

---

## 💡 Pro Tips

### Tip #1: Balance Precision vs Recall
```
High Precision (fewer false alarms):
→ Move threshold RIGHT (0.99+)

High Recall (catch more frauds):
→ Move threshold LEFT (0.95-)
```

### Tip #2: Focus on High-Value Frauds
```
Filter: Flagged Only = ON
Sort by: Amount (descending)
Review: Top 20 transactions
```

### Tip #3: Understand Why
```
Look at "Reasons" column:
• Shows which patterns triggered alert
• Feature breakdown in metadata
• Learn fraud signatures
```

### Tip #4: Test Edge Cases
```
Try:
• Very small amount (₹1)
• Very large amount (₹10M)
• Zero balance after
• Rapid succession (repeat=5)
```

### Tip #5: Export Results
```
File: flagged_transactions.csv
Contains all detected anomalies
Can be analyzed in Excel/Python
```

---

## 📞 Help Resources

### Quick Questions
1. **How does it work?**
   → Read: `QUICK_START.md`

2. **What can I customize?**
   → Read: `README.md` Configuration section

3. **How accurate is it?**
   → Check: `STATUS.md` Performance section

4. **What's under the hood?**
   → Read: `PROJECT_SUMMARY.md`

### Common Questions

**Q: Why unsupervised learning?**
A: Fraud labels are scarce, late, and outdated. Model normal behavior instead.

**Q: Why ensemble (IF + AE)?**
A: Different models catch different patterns. Together = more robust.

**Q: What's a good threshold?**
A: Depends on your use case. Default (0.977) balances precision/recall.

**Q: Can I use my own data?**
A: Yes! Update `data_path` in `app.py` CONFIG. Ensure CSV has similar columns.

**Q: How to deploy in production?**
A: Use WSGI server (gunicorn), database, authentication. See roadmap in docs.

---

## 🎓 Learning Path

### Beginner (1 hour)
1. Open http://127.0.0.1:5000
2. Start stream and watch
3. Read QUICK_START.md
4. Test 3-5 custom transactions

### Intermediate (3 hours)
5. Run `python main.py`
6. Examine generated plots
7. Read README.md fully
8. Modify threshold in code
9. Test different configurations

### Advanced (1 day)
10. Read PROJECT_SUMMARY.md
11. Study `anomaly_core.py` code
12. Modify feature engineering
13. Tune model hyperparameters
14. Integrate your own data

---

## ⚡ Cheat Sheet

### System Status
```bash
# Check if running
# Open: http://127.0.0.1:5000
# Should see dashboard

# Stop server
# Terminal: Ctrl+C
```

### Performance Targets
```
✅ ROC-AUC > 0.90
✅ Precision > 50%
✅ Recall > 60%
✅ Processing > 50 txn/sec
```

### File Locations
```
Code: GenAI/*.py
UI: GenAI/templates/*.html
Docs: GenAI/*.md
Data: data/*.csv
Plots: GenAI/plots/*.png
```

### API Endpoints
```
GET  /               Dashboard
GET  /api/stats      Statistics
POST /api/stream_batch  Get batch
POST /api/score_transaction  Score one
POST /api/update_threshold  Change threshold
```

---

## 🎯 Success Checklist

- [ ] Server running (http://127.0.0.1:5000 loads)
- [ ] Stream starts when clicked
- [ ] Transactions appear in feed
- [ ] Stats update in real-time
- [ ] Threshold slider works
- [ ] Custom transaction tested
- [ ] Understand why flags appear
- [ ] Read at least one doc file

---

## 🏆 You're Ready!

**You now know:**
✅ How to access the system  
✅ How to control the UI  
✅ How to interpret results  
✅ How to test transactions  
✅ How to tune performance  
✅ Where to get help  

**Go forth and detect fraud! 🕵️‍♂️**

---

**Quick Links:**
- 🌐 **App:** http://127.0.0.1:5000
- 📖 **Tutorial:** QUICK_START.md
- 📚 **Docs:** README.md
- 🔧 **Status:** STATUS.md

**Need Help?** Check the documentation files!
