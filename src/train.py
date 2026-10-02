
import os
import sys
import json
import time
import warnings
import joblib
 
import numpy as np
import pandas as pd
 
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import (
    StratifiedKFold, cross_validate,
    RandomizedSearchCV, train_test_split
)
from sklearn.metrics import (
    roc_auc_score, f1_score, average_precision_score,
    classification_report, roc_curve, precision_recall_curve
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
 
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
 
warnings.filterwarnings('ignore')
 
# ── Column definitions (mirrors preprocess.py) ────────────────────────────
NUMERIC_COLS     = ['tenure', 'MonthlyCharges', 'TotalCharges',
                    'AvgMonthlyCharge', 'ServiceCount']
CATEGORICAL_COLS = ['MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport',
                    'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod']
BINARY_COLS      = ['gender', 'Partner', 'Dependents', 'PhoneService',
                    'PaperlessBilling', 'SeniorCitizen', 'HighRisk']
TARGET           = 'Churn'
 
 
# 1. DATA LOADING & CLEANING
 
def load_and_clean(path: str) -> tuple[pd.DataFrame, pd.Series]:
   
    df = pd.read_csv(path)
 
    # Fix 1: TotalCharges stored as string with whitespace for tenure=0 rows
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0.0)
 
    # Fix 2: Drop non-feature ID column
    df.drop(columns=['customerID'], inplace=True)
 
    # Fix 3: Encode target
    df[TARGET] = (df[TARGET] == 'Yes').astype(int)
 
    # Fix 4: Binary Yes/No → 1/0 (deterministic, not fitted)
    for col in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
 
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y
 

# 2. FEATURE ENGINEERING

def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    
    X = X.copy()
 
    X['AvgMonthlyCharge'] = X.apply(
        lambda r: r['TotalCharges'] / r['tenure']
        if r['tenure'] > 0 else r['MonthlyCharges'],
        axis=1
    )
 
    service_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    X['ServiceCount'] = sum((X[c] == 'Yes').astype(int) for c in service_cols)
 
    X['HighRisk'] = (
        (X['Contract'] == 'Month-to-month') &
        (X['tenure'] < 12) &
        (X['PaymentMethod'] == 'Electronic check')
    ).astype(int)
 
    return X
 

# 3. PREPROCESSOR (fitted on train only)

 
def build_preprocessor() -> ColumnTransformer:
    
    return ColumnTransformer(
        transformers=[
            ('num', StandardScaler(),                                         NUMERIC_COLS),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CATEGORICAL_COLS),
            ('bin', 'passthrough',                                            BINARY_COLS),
        ],
        remainder='drop'
    )
 
 # 4. CROSS-VALIDATION HELPER

 
def cv_evaluate(pipeline, X, y, n_splits=5, label='Model') -> dict:
    """5-fold stratified CV — returns mean ± std for 3 metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_validate(
        pipeline, X, y, cv=skf,
        scoring={'roc_auc': 'roc_auc', 'f1': 'f1', 'pr_auc': 'average_precision'}
    )
    result = {
        'ROC-AUC': f"{scores['test_roc_auc'].mean():.4f} ± {scores['test_roc_auc'].std():.4f}",
        'F1':      f"{scores['test_f1'].mean():.4f} ± {scores['test_f1'].std():.4f}",
        'PR-AUC':  f"{scores['test_pr_auc'].mean():.4f} ± {scores['test_pr_auc'].std():.4f}",
        # Raw means for comparison
        '_roc_auc_mean': scores['test_roc_auc'].mean(),
        '_f1_mean':      scores['test_f1'].mean(),
        '_pr_auc_mean':  scores['test_pr_auc'].mean(),
    }
    print(f"\n  {label}")
    print(f"    ROC-AUC : {result['ROC-AUC']}")
    print(f"    F1      : {result['F1']}")
    print(f"    PR-AUC  : {result['PR-AUC']}")
    return result
 
 
# 5. EVALUATION PLOTS
 
def save_evaluation_plots(model, X_test, y_test, out_dir='../models'):
    """Saves ROC curve and Precision-Recall curve for the best model."""
    probs = model.predict_proba(X_test)[:, 1]
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
 
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    axes[0].plot(fpr, tpr, color='#E8704C', lw=2, label=f'AUC = {auc:.4f}')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    axes[0].fill_between(fpr, tpr, alpha=0.1, color='#E8704C')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve — Best Model (Holdout)', fontweight='bold')
    axes[0].legend(loc='lower right')
 
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, probs)
    pr_auc = average_precision_score(y_test, probs)
    baseline = y_test.mean()
    axes[1].plot(recall, precision, color='#4C9BE8', lw=2, label=f'PR-AUC = {pr_auc:.4f}')
    axes[1].axhline(baseline, color='gray', linestyle='--', lw=1, label=f'Baseline = {baseline:.2f}')
    axes[1].fill_between(recall, precision, alpha=0.1, color='#4C9BE8')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve — Best Model (Holdout)', fontweight='bold')
    axes[1].legend(loc='upper right')
 
    plt.tight_layout()
    plot_path = os.path.join(out_dir, 'evaluation_curves.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Evaluation plots saved → {plot_path}")
 
 

# 6. MAIN TRAINING PIPELINE
 
def main():
    # ── Paths ─────────────────────────────────────────────────────────────
    data_path  = os.path.join(os.path.dirname(__file__), '..', 'data',
                              'WA_Fn-UseC_-Telco-Customer-Churn.csv')
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
 
    print("=" * 60)
    print("  TELCO CHURN — MODEL TRAINING PIPELINE")
    print("=" * 60)
 
    # ── Load & engineer features ──────────────────────────────────────────
    print("\n[1/5] Loading and cleaning data...")
    X, y = load_and_clean(data_path)
    X = engineer_features(X)
    print(f"      Shape: {X.shape}  |  Churn rate: {y.mean()*100:.1f}%")
 
    # ── Stratified holdout split (FIRST step — test set locked away) ──────
    print("\n[2/5] Splitting data (80% train / 20% holdout test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"      Train: {X_train.shape[0]} rows  |  Test: {X_test.shape[0]} rows")
    print(f"      Train churn rate: {y_train.mean()*100:.1f}%  |  Test: {y_test.mean()*100:.1f}%")
    print("      ✓ No preprocessing has touched the test set yet.")
 
    pre = build_preprocessor()
    all_metrics = {}
 
    # MODEL 1: Logistic Regression (Baseline)
    print("\n[3/5] Training models...")
    print("\n  ── Baseline: Logistic Regression ──")
 
    lr_pipe = Pipeline([
        ('pre', build_preprocessor()),
        ('clf', LogisticRegression(
            class_weight='balanced',
            max_iter=1000,
            C=1.0,
            solver='lbfgs',
            random_state=42
        ))
    ])
    all_metrics['1_LogisticRegression_Baseline'] = cv_evaluate(
        lr_pipe, X_train, y_train, label='Logistic Regression (baseline)'
    )
 
    
    # MODEL 2: Random Forest
    print("\n  ── Random Forest ──")
 
    rf_pipe = Pipeline([
        ('pre', build_preprocessor()),
        ('clf', RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])
    all_metrics['2_RandomForest'] = cv_evaluate(
        rf_pipe, X_train, y_train, label='Random Forest'
    )
 
    
    # MODEL 3: GradientBoosting — Baseline then Tuned
    print("\n  ── GradientBoosting (default) ──")
 
    gb_pipe = Pipeline([
        ('pre', build_preprocessor()),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])
    all_metrics['3_GradientBoosting_Default'] = cv_evaluate(
        gb_pipe, X_train, y_train, label='GradientBoosting (default params)'
    )
 
    # MODEL 4: GradientBoosting — RandomizedSearchCV Tuning
    print("\n  ── GradientBoosting (RandomizedSearchCV tuning) ──")
    print("     Running 15 candidates × 3-fold CV (this takes ~2 minutes)...")
 
    gb_tune_pipe = Pipeline([
        ('pre', build_preprocessor()),
        ('clf', GradientBoostingClassifier(random_state=42))
    ])
 
    param_dist = {
        'clf__n_estimators':     [100, 200, 300],
        'clf__max_depth':        [3, 4, 5],
        'clf__learning_rate':    [0.05, 0.1, 0.15],
        'clf__subsample':        [0.7, 0.8, 1.0],
        'clf__min_samples_leaf': [10, 20, 30],
    }
 
    t_start = time.time()
    search = RandomizedSearchCV(
        gb_tune_pipe, param_dist,
        n_iter=15,
        scoring='roc_auc',
        cv=StratifiedKFold(3, shuffle=True, random_state=42),
        random_state=42, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"     Tuning done in {time.time()-t_start:.0f}s")
    print(f"     Best params: {search.best_params_}")
    print(f"     Best CV ROC-AUC: {search.best_score_:.4f}")
 
    best_pipe = search.best_estimator_
 
    # 5-fold CV on best params
    all_metrics['4_GradientBoosting_Tuned'] = cv_evaluate(
        best_pipe, X_train, y_train, label='GradientBoosting (tuned) — 5-fold CV'
    )
    all_metrics['4_GradientBoosting_Tuned']['best_params'] = search.best_params_
 
    # 5. HOLDOUT TEST SET EVALUATION
    print("\n[4/5] Evaluating best model on holdout test set...")
 
    # Fit best pipeline on full training set
    best_pipe.fit(X_train, y_train)
 
    probs = best_pipe.predict_proba(X_test)[:, 1]
    preds = best_pipe.predict(X_test)
 
    holdout_metrics = {
        'ROC-AUC': round(roc_auc_score(y_test, probs), 4),
        'F1':      round(f1_score(y_test, preds), 4),
        'PR-AUC':  round(average_precision_score(y_test, probs), 4),
    }
    all_metrics['5_BestModel_HoldoutTest'] = holdout_metrics
 
    print(f"\n  Holdout Test Results (n={len(y_test)}, never seen during training):")
    print(f"    ROC-AUC : {holdout_metrics['ROC-AUC']}")
    print(f"    F1      : {holdout_metrics['F1']}")
    print(f"    PR-AUC  : {holdout_metrics['PR-AUC']}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, preds, target_names=['No Churn', 'Churn']))
 
    save_evaluation_plots(best_pipe, X_test, y_test, out_dir=models_dir)
 
    # 6. FINAL MODEL — REFIT ON ALL DATA & SAVE
    print("\n[5/5] Fitting final model on full dataset and saving...")
    best_pipe.fit(X, y)   # refit on 100% data now that evaluation is done
 
    model_path   = os.path.join(models_dir, 'best_model.joblib')
    metrics_path = os.path.join(models_dir, 'metrics_summary.json')
 
    joblib.dump(best_pipe, model_path)
 
    # Clean metrics for JSON (remove internal _keys)
    clean_metrics = {
        k: {mk: mv for mk, mv in v.items() if not mk.startswith('_')}
        for k, v in all_metrics.items()
    }
    with open(metrics_path, 'w') as f:
        json.dump(clean_metrics, f, indent=2)
 
    # SUMMARY TABLE
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON SUMMARY (5-fold CV on training set)")
    print("=" * 60)
    print(f"  {'Model':<35} {'ROC-AUC':>12} {'F1':>12} {'PR-AUC':>12}")
    print("  " + "-" * 74)
 
    model_labels = {
        '1_LogisticRegression_Baseline': 'Logistic Regression (baseline)',
        '2_RandomForest':                'Random Forest (300 trees)',
        '3_GradientBoosting_Default':    'GradientBoosting (default)',
        '4_GradientBoosting_Tuned':      'GradientBoosting (tuned) ⭐',
    }
    for key, label in model_labels.items():
        m = all_metrics[key]
        roc = m['ROC-AUC'] if isinstance(m['ROC-AUC'], str) else str(m['ROC-AUC'])
        f1  = m['F1']      if isinstance(m['F1'], str)      else str(m['F1'])
        pr  = m['PR-AUC']  if isinstance(m['PR-AUC'], str)  else str(m['PR-AUC'])
        print(f"  {label:<35} {roc:>12} {f1:>12} {pr:>12}")
 
    print("  " + "-" * 74)
    h = all_metrics['5_BestModel_HoldoutTest']
    print(f"  {'Holdout Test (best model)':<35} {str(h['ROC-AUC']):>12} {str(h['F1']):>12} {str(h['PR-AUC']):>12}")
 
    print(f"\n  Model saved  → {model_path}")
    print(f"  Metrics saved → {metrics_path}")
    print("\n  ✓ Training complete. Run predict.py or start the API next.")
    print("=" * 60)
 
 
if __name__ == '__main__':
    main()
 

