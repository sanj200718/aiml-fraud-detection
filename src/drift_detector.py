import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def calculate_psi(baseline: np.ndarray, current: np.ndarray, num_buckets: int = 10) -> float:
    """
    Calculates Population Stability Index (PSI) between baseline and production window.
    PSI < 0.10: No significant shift
    0.10 <= PSI <= 0.25: Moderate shift
    PSI > 0.25: Substantial drift (Trigger Retrain)
    """
    baseline = baseline[~np.isnan(baseline)]
    current = current[~np.isnan(current)]
    
    percentiles = np.linspace(0, 100, num_buckets + 1)
    buckets = np.percentile(baseline, percentiles)
    buckets[0] -= 1e-5
    buckets[-1] += 1e-5

    base_counts, _ = np.histogram(baseline, bins=buckets)
    curr_counts, _ = np.histogram(current, bins=buckets)

    base_pct = base_counts / len(baseline)
    curr_pct = curr_counts / len(current)

    # Replace zero percentages to avoid division by zero
    base_pct = np.where(base_pct == 0, 1e-4, base_pct)
    curr_pct = np.where(curr_pct == 0, 1e-4, curr_pct)

    psi_val = np.sum((curr_pct - base_pct) * np.log(curr_pct / base_pct))
    return float(psi_val)

class DriftMonitor:
    """Monitors incoming transaction windows and triggers retrain alerts."""
    def __init__(self, psi_threshold: float = 0.25, ks_alpha: float = 0.05):
        self.psi_threshold = psi_threshold
        self.ks_alpha = ks_alpha

    def inspect_window(self, baseline_df: pd.DataFrame, current_df: pd.DataFrame, features: list[str]) -> dict:
        drift_report = {}
        flagged_retrain = False

        for col in features:
            base_col = baseline_df[col].values
            curr_col = current_df[col].values

            psi = calculate_psi(base_col, curr_col)
            ks_stat, p_value = ks_2samp(base_col, curr_col)

            is_drifted = (psi > self.psi_threshold) or (p_value < self.ks_alpha)
            if is_drifted:
                flagged_retrain = True

            drift_report[col] = {
                'psi': round(psi, 4),
                'ks_stat': round(ks_stat, 4),
                'p_value': round(p_value, 5),
                'drift_detected': is_drifted
            }

        return {
            'flag_retrain': flagged_retrain,
            'feature_reports': drift_report
        }