import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.data_loader import generate_synthetic_credit_card_data, simulate_concept_drift
from src.models import SupervisedFraudModel, UnsupervisedAnomalyModel
from src.evaluation import compute_pr_auc, evaluate_cost_matrix
from src.drift_detector import DriftMonitor

def run_pipeline():
    print("=== [1/5] Generating Synthetic Credit Card Dataset ===")
    raw_df = generate_synthetic_credit_card_data(n_samples=80000, fraud_ratio=0.0017)
    
    feature_cols = [col for col in raw_df.columns if col not in ['Time', 'Class']]
    X = raw_df[feature_cols].values
    y = raw_df['Class'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    print(f"Train Set: {len(X_train)} rows | Fraud cases: {int(y_train.sum())}")
    print(f"Test Set:  {len(X_test)} rows  | Fraud cases: {int(y_test.sum())}")

    print("\n=== [2/5] Training Supervised & Unsupervised Models ===")
    # Supervised XGBoost with imbalance weighting
    scale_pos = (len(y_train) - y_train.sum()) / y_train.sum()
    sup_model = SupervisedFraudModel(scale_pos_weight=scale_pos)
    sup_model.fit(X_train, y_train)
    sup_scores = sup_model.predict_proba(X_test)

    # Unsupervised Isolation Forest
    unsup_model = UnsupervisedAnomalyModel(contamination=0.0017)
    unsup_model.fit(X_train[y_train == 0]) # Train on normal transactions only
    unsup_scores = unsup_model.predict_anomaly_score(X_test)

    print("\n=== [3/5] Evaluating Models (PR-AUC Benchmark) ===")
    sup_pr_auc, _, _ = compute_pr_auc(y_test, sup_scores)
    unsup_pr_auc, _, _ = compute_pr_auc(y_test, unsup_scores)

    print(f"--> Supervised XGBoost PR-AUC:       {sup_pr_auc:.4f}")
    print(f"--> Unsupervised Isolation Forest PR-AUC: {unsup_pr_auc:.4f}")

    print("\n=== [4/5] Financial Cost Matrix Optimization ===")
    cost_analysis = evaluate_cost_matrix(y_test, sup_scores, cost_fp=20.0, cost_fn=250.0)
    print(f"--> Optimal Classification Threshold: {cost_analysis['best_threshold']:.2f}")
    print(f"--> Minimum Estimated Loss:           ${cost_analysis['min_total_cost']:,.2f}")

    print("\n=== [5/5] Simulating Concept Drift & Drift Alerting ===")
    baseline_df = pd.DataFrame(X_train, columns=feature_cols)
    normal_production_df = pd.DataFrame(X_test, columns=feature_cols)
    drifted_production_df = simulate_concept_drift(normal_production_df, shift_intensity=1.8)

    monitor = DriftMonitor(psi_threshold=0.25)

    print("\n--- Test 1: Baseline vs Clean Production Window ---")
    clean_report = monitor.inspect_window(baseline_df, normal_production_df, feature_cols)
    print(f"Retrain Trigger Flagged: {clean_report['flag_retrain']}")

    print("\n--- Test 2: Baseline vs Shifted/Drifted Production Window ---")
    drift_report = monitor.inspect_window(baseline_df, drifted_production_df, feature_cols)
    print(f"Retrain Trigger Flagged: {drift_report['flag_retrain']}")

    drifted_feats = [col for col, res in drift_report['feature_reports'].items() if res['drift_detected']]
    print(f"Features triggering alerts: {drifted_feats}")

if __name__ == "__main__":
    run_pipeline()