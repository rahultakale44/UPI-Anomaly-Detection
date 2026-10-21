"""Flask app with real-time transaction streaming and anomaly detection.
Run: python app.py -> open http://127.0.0.1:5000"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import os, time, json
from flask import Flask, jsonify, request, render_template, Response
from anomaly_core import *

app = Flask(__name__)

# Global state
DATA = None
DETECTOR = None
TRAIN_DATA = None
CONFIG = {
    'data_path': '../data/PS_20174392719_1491204439457_log.csv',
    'train_sample': 50000,  # Use 50k transactions for training
    'stream_batch': 100,    # Stream 100 transactions at a time
    'current_index': 0
}

def initialize_system():
    """Load data and train detector."""
    global DATA, DETECTOR, TRAIN_DATA
    
    if not os.path.exists(CONFIG['data_path']):
        print(f"ERROR: Data file not found at {CONFIG['data_path']}")
        return False
    
    # Load full dataset
    print("\n" + "="*70)
    print("INITIALIZING ANOMALY DETECTION SYSTEM")
    print("="*70)
    
    DATA = load_real_data(CONFIG['data_path'])
    DATA = build_features(DATA)
    
    # Use first N transactions for training (only non-fraud)
    train_mask = (DATA.index < CONFIG['train_sample']) & (DATA['isFraud'] == 0)
    TRAIN_DATA = DATA[train_mask].copy()
    
    print(f"\nTraining on {len(TRAIN_DATA):,} normal transactions...")
    X_train = TRAIN_DATA[FEATURES].values
    
    DETECTOR = Detector(alert_rate=0.02, seed=42).fit(X_train)
    
    print("\n" + "="*70)
    print("SYSTEM READY")
    print("="*70 + "\n")
    
    return True

def format_transaction(row, score_info):
    """Format a single transaction for frontend."""
    return {
        'step': int(row.step),
        'type': row.type,
        'amount': float(row.amount),
        'nameOrig': row.nameOrig,
        'nameDest': row.nameDest,
        'oldbalanceOrg': float(row.oldbalanceOrg),
        'newbalanceOrig': float(row.newbalanceOrig),
        'score': float(score_info['score']),
        'iso_score': float(score_info['iso']),
        'ae_score': float(score_info['ae']),
        'ae_top_feature': FEATURES[int(np.argmax(score_info['ae_feat']))],
        'reasons': explain(row),
        'is_fraud': int(row.isFraud),
        'is_flagged': int(row.isFlaggedFraud)
    }

@app.route("/")
def index():
    """Serve main UI."""
    if DATA is None:
        return "System not initialized. Check console for errors.", 500
    
    return render_template(
        "index.html",
        threshold=float(DETECTOR.threshold),
        total_transactions=len(DATA),
        fraud_count=int(DATA['isFraud'].sum()),
        types=list(DATA['type'].unique())
    )

@app.route("/api/stats")
def stats():
    """Get dataset statistics."""
    if DATA is None:
        return jsonify({'error': 'System not initialized'}), 500
    
    return jsonify({
        'total': len(DATA),
        'fraud_count': int(DATA['isFraud'].sum()),
        'fraud_rate': float(DATA['isFraud'].mean()),
        'types': DATA['type'].value_counts().to_dict(),
        'current_index': CONFIG['current_index'],
        'threshold': float(DETECTOR.threshold)
    })

@app.route("/api/stream")
def stream():
    """Server-Sent Events endpoint for real-time streaming."""
    def generate():
        idx = CONFIG['current_index']
        batch_size = CONFIG['stream_batch']
        
        while idx < len(DATA):
            # Get next batch
            batch_end = min(idx + batch_size, len(DATA))
            batch = DATA.iloc[idx:batch_end]
            
            # Score batch
            X_batch = batch[FEATURES].values
            scores = DETECTOR.score(X_batch)
            
            # Format transactions
            transactions = []
            for i, (_, row) in enumerate(batch.iterrows()):
                score_info = {
                    'score': scores['score'][i],
                    'iso': scores['iso'][i],
                    'ae': scores['ae'][i],
                    'ae_feat': scores['ae_feat'][i]
                }
                transactions.append(format_transaction(row, score_info))
            
            # Send to client
            yield f"data: {json.dumps({'transactions': transactions})}\n\n"
            
            idx = batch_end
            CONFIG['current_index'] = idx
            time.sleep(0.1)  # Small delay for visualization
        
        yield f"data: {json.dumps({'done': True})}\n\n"
    
    return Response(generate(), mimetype='text/event-stream')

@app.post("/api/reset_stream")
def reset_stream():
    """Reset stream to beginning."""
    CONFIG['current_index'] = 0
    return jsonify({'success': True})

@app.post("/api/stream_batch")
def stream_batch():
    """Get a batch of transactions (alternative to SSE)."""
    params = request.get_json(force=True)
    start_idx = int(params.get('start', CONFIG['current_index']))
    batch_size = int(params.get('batch_size', CONFIG['stream_batch']))
    
    if start_idx >= len(DATA):
        return jsonify({'transactions': [], 'done': True, 'next_index': start_idx})
    
    batch_end = min(start_idx + batch_size, len(DATA))
    batch = DATA.iloc[start_idx:batch_end]
    
    # Score batch
    X_batch = batch[FEATURES].values
    scores = DETECTOR.score(X_batch)
    
    # Format transactions
    transactions = []
    for i, (_, row) in enumerate(batch.iterrows()):
        score_info = {
            'score': scores['score'][i],
            'iso': scores['iso'][i],
            'ae': scores['ae'][i],
            'ae_feat': scores['ae_feat'][i]
        }
        transactions.append(format_transaction(row, score_info))
    
    CONFIG['current_index'] = batch_end
    
    return jsonify({
        'transactions': transactions,
        'done': batch_end >= len(DATA),
        'next_index': batch_end,
        'total': len(DATA)
    })

@app.post("/api/score_transaction")
def score_transaction():
    """Score a custom transaction."""
    params = request.get_json(force=True)
    
    # Create transaction dataframe
    txn = pd.DataFrame([{
        'step': int(params.get('step', DATA['step'].max() + 1)),
        'type': params['type'],
        'amount': float(params['amount']),
        'nameOrig': params['nameOrig'],
        'oldbalanceOrg': float(params.get('oldbalanceOrg', 0)),
        'newbalanceOrig': float(params.get('newbalanceOrig', 0)),
        'nameDest': params['nameDest'],
        'oldbalanceDest': float(params.get('oldbalanceDest', 0)),
        'newbalanceDest': float(params.get('newbalanceDest', 0)),
        'isFraud': 0,
        'isFlaggedFraud': 0
    }])
    
    # Build features
    txn = build_features(txn)
    X = txn[FEATURES].values
    scores = DETECTOR.score(X)
    
    score_info = {
        'score': scores['score'][0],
        'iso': scores['iso'][0],
        'ae': scores['ae'][0],
        'ae_feat': scores['ae_feat'][0]
    }
    
    result = format_transaction(txn.iloc[0], score_info)
    result['is_flagged'] = int(result['score'] >= DETECTOR.threshold)
    
    return jsonify(result)

@app.post("/api/update_threshold")
def update_threshold():
    """Update detection threshold."""
    params = request.get_json(force=True)
    DETECTOR.threshold = float(params['threshold'])
    return jsonify({'threshold': float(DETECTOR.threshold)})

if __name__ == "__main__":
    if initialize_system():
        print("Starting Flask server on http://127.0.0.1:5000")
        app.run(debug=False, threaded=True)
    else:
        print("Failed to initialize system. Please check data path.")
