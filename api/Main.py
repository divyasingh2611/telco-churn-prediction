from pathlib import Path
from typing import Union, Optional
 
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator, model_validator
 
# ── App setup ─────────────────────────────────────────────────────────────
app = FastAPI(
    title='Telco Churn Prediction API',
    description=(
        'Predicts whether a telecom customer will churn, '
        'returning a binary prediction and probability score.'
    ),
    version='1.0.0',
)
 
# ── Model loading (once at startup, not per request) ──────────────────────
MODEL_PATH = Path(__file__).resolve().parent.parent / 'models' / 'best_model.joblib'
_model = None
 
 
def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise RuntimeError(
                f"Model file not found at {MODEL_PATH}. "
                "Run `python src/train.py` first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model
 
 
@app.on_event('startup')
def load_model_on_startup():
    """Pre-load model when the server starts — avoids cold-start latency."""
    get_model()
 
 
# ── Preprocessing constants (must mirror train.py) ────────────────────────
BINARY_YES_NO = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
SERVICE_COLS  = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                 'TechSupport', 'StreamingTV', 'StreamingMovies']
 
 
# ── Request schema ────────────────────────────────────────────────────────
class CustomerProfile(BaseModel):
    
    customerID:       Optional[str]        = None
    gender:           str
    SeniorCitizen:    int
    Partner:          str
    Dependents:       str
    tenure:           int
    PhoneService:     str
    MultipleLines:    str
    InternetService:  str
    OnlineSecurity:   str
    OnlineBackup:     str
    DeviceProtection: str
    TechSupport:      str
    StreamingTV:      str
    StreamingMovies:  str
    Contract:         str
    PaperlessBilling: str
    PaymentMethod:    str
    MonthlyCharges:   float
    TotalCharges:     Union[float, str]    # handles "29.85" or 29.85
 
    @field_validator('TotalCharges', mode='before')
    @classmethod
    def parse_total_charges(cls, v):
        """Coerces TotalCharges to float; returns 0.0 for blank/invalid."""
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0
 
    @field_validator('tenure')
    @classmethod
    def tenure_non_negative(cls, v):
        if v < 0:
            raise ValueError('tenure must be >= 0')
        return v
 
    @field_validator('SeniorCitizen')
    @classmethod
    def senior_citizen_binary(cls, v):
        if v not in (0, 1):
            raise ValueError('SeniorCitizen must be 0 or 1')
        return v
 
 
# ── Response schema ───────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    customerID:        Optional[str]
    churn_prediction:  int            # 0 = No, 1 = Yes
    churn_label:       str            # "Yes" or "No"
    churn_probability: float          # raw probability score [0.0, 1.0]
    risk_tier:         str            # "LOW" / "MEDIUM" / "HIGH"
 
 
# ── Preprocessing helper ──────────────────────────────────────────────────
def preprocess_single(data: dict) -> pd.DataFrame:
    """
    Applies deterministic preprocessing to a single customer dict.
    Mirrors the exact steps in train.py's load_and_clean + engineer_features.
    """
    df = pd.DataFrame([data])
 
    df.drop(columns=[c for c in ['customerID'] if c in df.columns], inplace=True)
 
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
 
    for col in BINARY_YES_NO:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
 
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
 
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
    if prob >= 0.70:
        return 'HIGH'
    elif prob >= 0.40:
        return 'MEDIUM'
    return 'LOW'
 
 
# ── Routes ────────────────────────────────────────────────────────────────
@app.get('/', tags=['Info'])
def root():
    return {
        'service': 'Telco Churn Prediction API',
        'version': '1.0.0',
        'docs':    '/docs',
        'health':  '/health',
        'predict': 'POST /predict',
    }
 
 
@app.get('/health', tags=['Info'])
def health():
    """Returns model load status. Use this to verify the server is ready."""
    try:
        model = get_model()
        return {
            'status':       'ok',
            'model_loaded': True,
            'model_path':   str(MODEL_PATH),
        }
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
 
 
@app.post('/predict', response_model=PredictionResponse, tags=['Inference'])
def predict(customer: CustomerProfile):
    """
    Accepts a raw customer JSON payload and returns:
    - churn_prediction: 0 or 1
    - churn_label: "No" or "Yes"
    - churn_probability: float between 0 and 1
    - risk_tier: LOW / MEDIUM / HIGH
    """
    try:
        model = get_model()
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
 
    raw = customer.model_dump()
    customer_id = raw.pop('customerID', None)
 
    df   = preprocess_single(raw)
    prob = float(model.predict_proba(df)[0][1])
    pred = int(prob >= 0.5)
 
    return PredictionResponse(
        customerID=customer_id,
        churn_prediction=pred,
        churn_label='Yes' if pred == 1 else 'No',
        churn_probability=round(prob, 4),
        risk_tier=risk_tier(prob),
    )
 