import argparse
import json
import sys
from pathlib import Path
 
import joblib
import pandas as pd
 
# ── Model path (relative to this file) ───────────────────────────────────
MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'best_model.joblib'
 
# ── Preprocessing constants (must match train.py exactly) ─────────────────
BINARY_YES_NO = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
SERVICE_COLS  = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                 'TechSupport', 'StreamingTV', 'StreamingMovies']
 
 
def load_model():
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model not found at: {MODEL_PATH}", file=sys.stderr)
        print("        Run `python train.py` first to generate the model.", file=sys.stderr)
        sys.exit(1)
    return joblib.load(MODEL_PATH)
 
 
def preprocess_single(raw: dict) -> pd.DataFrame:
    """
    Applies the same deterministic preprocessing as train.py to a single
    raw JSON payload. No model fitting — pure transformation.
    """
    df = pd.DataFrame([raw])
 
    # Drop ID (not a feature)
    df.drop(columns=[c for c in ['customerID'] if c in df.columns], inplace=True)
 
    # Fix TotalCharges: may arrive as string (e.g. "29.85") or whitespace
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
 
    # Binary encode Yes/No columns
    for col in BINARY_YES_NO:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
 
    # Binary encode gender
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
 
    # Feature engineering (mirrors train.py)
    df['AvgMonthlyCharge'] = df.apply(
        lambda r: r['TotalCharges'] / r['tenure']
        if r['tenure'] > 0 else r['MonthlyCharges'],
        axis=1
    )
    df['ServiceCount'] = sum(
        (df[c] == 'Yes').astype(int) for c in SERVICE_COLS if c in df.columns
    )
    df['HighRisk'] = (
        (df['Contract'] == 'Month-to-month') &
        (df['tenure'] < 12) &
        (df['PaymentMethod'] == 'Electronic check')
    ).astype(int)
 
    return df
 
 
def risk_tier(prob: float) -> str:
    """Converts probability to a human-readable risk tier."""
    if prob >= 0.70:
        return 'HIGH'
    elif prob >= 0.40:
        return 'MEDIUM'
    return 'LOW'
 
 
def predict(raw_payload: dict) -> dict:
    """
    Full inference pipeline: raw JSON dict → prediction dict.
    Loads model, preprocesses, predicts.
    """
    model = load_model()
    df    = preprocess_single(raw_payload)
 
    prob = float(model.predict_proba(df)[0][1])
    pred = int(prob >= 0.5)
 
    return {
        'customerID':        raw_payload.get('customerID', 'N/A'),
        'churn_prediction':  pred,
        'churn_label':       'Yes' if pred == 1 else 'No',
        'churn_probability': round(prob, 4),
        'risk_tier':         risk_tier(prob),
    }
 
 
SAMPLE_PAYLOAD = {
    "customerID":       "7590-VHVEG",
    "gender":           "Female",
    "SeniorCitizen":    0,
    "Partner":          "Yes",
    "Dependents":       "No",
    "tenure":           1,
    "PhoneService":     "No",
    "MultipleLines":    "No phone service",
    "InternetService":  "DSL",
    "OnlineSecurity":   "No",
    "OnlineBackup":     "Yes",
    "DeviceProtection": "No",
    "TechSupport":      "No",
    "StreamingTV":      "No",
    "StreamingMovies":  "No",
    "Contract":         "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod":    "Electronic check",
    "MonthlyCharges":   29.85,
    "TotalCharges":     "29.85"
}
 
 
def main():
    parser = argparse.ArgumentParser(
        description='Telco Churn Prediction — CLI Inference',
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        '--input', type=str,
        help='Path to a JSON file containing a single customer profile'
    )
    parser.add_argument(
        '--json', type=str,
        help='Inline JSON string of a customer profile'
    )
    args = parser.parse_args()
 
    if args.input:
        with open(args.input, 'r') as f:
            payload = json.load(f)
    elif args.json:
        payload = json.loads(args.json)
    else:
        print("[INFO] No input provided — using assessment sample payload.\n")
        payload = SAMPLE_PAYLOAD
 
    result = predict(payload)
    print(json.dumps(result, indent=2))
 
 
if __name__ == '__main__':
    main()
 

