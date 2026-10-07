import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def generate_synthetic_credit_card_data(n_samples: int = 100000, fraud_ratio: float = 0.0017, random_state: int = 42) -> pd.DataFrame:
    """
    Generates synthetic transaction data replicating Kaggle's Credit Card Fraud dataset structure:
    - Time (seconds elapsed)
    - V1 to V28 (PCA transformed features)
    - Amount (Transaction dollar value)
    - Class (0: Legitimate, 1: Fraudulent)
    """
    np.random.seed(random_state)
    n_fraud = int(n_samples * fraud_ratio)
    n_legit = n_samples - n_fraud

    # Legitimate transactions
    legit_v = np.random.normal(loc=0.0, scale=1.0, size=(n_legit, 28))
    legit_amount = np.random.exponential(scale=88.0, size=(n_legit, 1))
    legit_time = np.sort(np.random.uniform(0, 172800, size=(n_legit, 1)))
    legit_class = np.zeros((n_legit, 1))

    # Fraudulent transactions (shifted distribution on key PCA components)
    fraud_v = np.random.normal(loc=0.0, scale=1.0, size=(n_fraud, 28))
    fraud_v[:, [0, 1, 2, 3, 10, 11, 13]] += np.random.choice([-2.5, 2.5], size=(n_fraud, 7))
    fraud_amount = np.random.exponential(scale=122.0, size=(n_fraud, 1))
    fraud_time = np.sort(np.random.uniform(0, 172800, size=(n_fraud, 1)))
    fraud_class = np.ones((n_fraud, 1))

    legit_data = np.hstack([legit_time, legit_v, legit_amount, legit_class])
    fraud_data = np.hstack([fraud_time, fraud_v, fraud_amount, fraud_class])

    full_data = np.vstack([legit_data, fraud_data])
    np.random.shuffle(full_data)

    columns = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount', 'Class']
    df = pd.DataFrame(full_data, columns=columns)
    df['Class'] = df['Class'].astype(int)
    return df

def simulate_concept_drift(df: pd.DataFrame, shift_intensity: float = 1.5, random_state: int = 101) -> pd.DataFrame:
    """
    Injects covariate & concept drift into a dataset slice:
    - Mean and variance shift in features V1-V5 (new spender demographics / attack patterns)
    - Higher purchase amounts in fraudulent transactions
    """
    np.random.seed(random_state)
    drifted_df = df.copy()

    # Apply distribution shift on specific PCA features
    drifted_df['V1'] += np.random.normal(loc=shift_intensity, scale=0.5, size=len(drifted_df))
    drifted_df['V2'] *= np.random.uniform(1.2, 1.8, size=len(drifted_df))
    drifted_df['V4'] -= np.random.normal(loc=shift_intensity * 0.8, scale=0.3, size=len(drifted_df))
    
    # Shift spending amounts
    drifted_df['Amount'] = drifted_df['Amount'] * 1.4 + np.random.exponential(scale=15.0, size=len(drifted_df))
    
    return drifted_df