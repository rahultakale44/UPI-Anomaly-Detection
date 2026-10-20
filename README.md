# UPIShield — Transaction Anomaly Detection

A demo-ready unsupervised anomaly detector for mobile-money transactions, with a local dashboard, retrospective metrics, and measured model inference latency.

## Dataset
Use the PaySim mobile-money transaction dataset: https://www.kaggle.com/datasets/ealaxi/paysim1
Download it manually in a browser (Kaggle may require sign-in), extract the CSV, and put it in `data/raw/`. The usual filename is `PS_20174392719_152120.csv`.

**Honest scope:** PaySim is a simulated mobile-money dataset, not a live Indian UPI feed. Streamlit's manual scoring form simulates a real-time scoring interaction. No banking rails or bank accounts are connected.

## Windows / PowerShell setup
```powershell
cd D:\UPIShield
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```
If PowerShell blocks activation, use:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## First run
1. Place the extracted PaySim CSV in `data/raw/`.
2. Inspect a sample:
   ```powershell
   python -m src.data_profile
   ```
3. Train/evaluate:
   ```powershell
   python -m src.train
   ```
4. Launch dashboard:
   ```powershell
   streamlit run src/app.py
   ```

## What is measured
The script uses a chronological 70/15/15 split, fits Isolation Forest on non-fraud rows in the training period, tunes the threshold by F1 on labelled validation data, and evaluates on the later test period. Labels are excluded from model features and used only for training-normal filtering and retrospective threshold/evaluation.

This is a **semi-supervised evaluation protocol around an unsupervised anomaly model**, not a completely label-free threshold selection. State this clearly in the presentation. Metrics are benchmark results against PaySim's simulation labels and do not establish real-world UPI performance.

Model-only scoring latency is measured on this machine using one-row scoring calls. It excludes data loading, API/network time, dashboard rendering and any GenAI response time. Throughput is the reciprocal of mean single-row scoring latency, not a load-tested production throughput benchmark.

## GenAI requirement (Phase 2)
The core detector intentionally works without an API key. For a reliable tomorrow demo, use a grounded explanation template based only on calculated feature evidence and the anomaly score. If the evaluator requires an LLM-generated explanation specifically, configure a provider key locally in `.env`; never commit secrets. Keep a deterministic fallback and measure LLM latency separately. Never claim a transaction is definitely fraudulent based on an anomaly score alone.

## Limitations
- PaySim has no true UPI device/location history, so location-change and per-account velocity features cannot be honestly engineered from absent fields.
- Account identifiers are not included in the initial model features.
- A threshold selected with labels is not fully unsupervised.
- Test metrics can change with dataset version, split, hardware and dependencies.
