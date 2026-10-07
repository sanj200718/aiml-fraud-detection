import numpy as np
from sklearn.ensemble import IsolationForest
import xgboost as xgb

class SupervisedFraudModel:
    """Supervised XGBoost Classifier with algorithmic class weighting."""
    def __init__(self, scale_pos_weight: float = 1.0, random_state: int = 42):
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

class UnsupervisedAnomalyModel:
    """Unsupervised Isolation Forest Model for zero-shot outlier detection."""
    def __init__(self, contamination: float = 0.002, random_state: int = 42):
        self.model = IsolationForest(
            n_estimators=100,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1
        )

    def fit(self, X: np.ndarray):
        self.model.fit(X)

    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        # Lower decision function value means more anomalous; invert for probability proxy
        scores = -self.model.decision_function(X)
        # Min-max normalization for alignment with probabilities
        min_s, max_s = scores.min(), scores.max()
        return (scores - min_s) / (max_s - min_s + 1e-8)