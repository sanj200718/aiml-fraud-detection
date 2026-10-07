import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, confusion_matrix

def compute_pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Calculates PR-AUC score and returns precision and recall arrays."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    pr_auc = auc(recall, precision)
    return pr_auc, precision, recall

def evaluate_cost_matrix(
    y_true: np.ndarray, 
    y_scores: np.ndarray, 
    cost_fp: float = 20.0, 
    cost_fn: float = 250.0
) -> dict:
    """
    Evaluates business cost tradeoff:
    - False Positive (FP): Customer friction, support tickets ($20 default)
    - False Negative (FN): Direct chargeback loss + fees ($250 default)
    """
    thresholds = np.linspace(0.01, 0.99, 100)
    best_threshold = 0.5
    min_total_cost = float('inf')
    cost_breakdown = []

    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        
        total_cost = (fp * cost_fp) + (fn * cost_fn)
        cost_breakdown.append({
            'threshold': t,
            'fp': fp,
            'fn': fn,
            'tp': tp,
            'total_cost': total_cost
        })

        if total_cost < min_total_cost:
            min_total_cost = total_cost
            best_threshold = t

    return {
        'best_threshold': best_threshold,
        'min_total_cost': min_total_cost,
        'cost_curve': pd.DataFrame(cost_breakdown)
    }