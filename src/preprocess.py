import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
 
 
# ── Feature groups (from EDA) ─────────────────────────────────────────────
NUMERIC_FEATURES     = ['tenure', 'MonthlyCharges', 'TotalCharges']
ENGINEERED_NUMERIC   = ['AvgMonthlyCharge', 'ServiceCount', 'HighRisk']
BINARY_FEATURES      = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
CATEGORICAL_FEATURES = [
    'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
    'Contract', 'PaymentMethod'
]
DROP_COLS = ['customerID']
TARGET    = 'Churn'
 
 
class DataCleaner(BaseEstimator, TransformerMixin):
   
 
    BINARY_MAP = {
        'Yes': 1, 'No': 0,
        'Male': 1, 'Female': 0,
    }
 
    def fit(self, X, y=None):
        return self  # stateless — no leakage risk
 
    def transform(self, X):
        X = X.copy()
 
        # Fix TotalCharges: whitespace strings → NaN → 0.0
        # (EDA confirmed: all NaN rows have tenure=0, i.e., brand-new customers)
        X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0.0)
 
        # Drop non-feature columns
        X.drop(columns=[c for c in DROP_COLS if c in X.columns], inplace=True)
 
        # Binary encoding: Yes/No/Male/Female → 1/0
        for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'gender']:
            if col in X.columns:
                X[col] = X[col].map(self.BINARY_MAP).fillna(X[col])
 
        return X
 
 
class FeatureEngineer(BaseEstimator, TransformerMixin):
    
 
    SERVICE_COLS = [
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
        'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
 
    def fit(self, X, y=None):
        return self  # stateless — no leakage risk
 
    def transform(self, X):
        X = X.copy()
 
        # 1. Average monthly charge (handles tenure=0 safely)
        X['AvgMonthlyCharge'] = X.apply(
            lambda r: r['TotalCharges'] / r['tenure']
            if r['tenure'] > 0 else r['MonthlyCharges'],
            axis=1
        )
 
        # 2. Service bundle count
        X['ServiceCount'] = sum(
            (X[col] == 'Yes').astype(int) for col in self.SERVICE_COLS
            if col in X.columns
        )
 
        # 3. High-risk composite flag (EDA: 42%+ churn for this segment)
        X['HighRisk'] = (
            (X['Contract'] == 'Month-to-month') &
            (X['tenure'] < 12) &
            (X['PaymentMethod'] == 'Electronic check')
        ).astype(int)
 
        return X
 
 
def build_preprocessor():
    
    all_numeric = NUMERIC_FEATURES + ['AvgMonthlyCharge', 'ServiceCount']
    passthrough  = BINARY_FEATURES + ['SeniorCitizen', 'HighRisk']
 
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
 
    categorical_transformer = Pipeline([
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
 
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer,     all_numeric),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
            ('bin', 'passthrough',           passthrough),
        ],
        remainder='drop'  # explicitly drop anything not listed
    )
 
    return preprocessor
 

