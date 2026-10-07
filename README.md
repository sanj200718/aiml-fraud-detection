# Fraud & Anomaly Detection with Drift Monitoring Engine (AIML-02)

## Overview
A comprehensive Machine Learning pipeline designed for extreme class imbalance (<0.2% fraud) in credit card transactions. The project compares supervised and unsupervised models using Precision-Recall metrics, optimizes financial cost thresholds, and features a automated Population Stability Index (PSI) drift monitoring engine that flags model degradation for automatic retraining.

## Project Architecture & Methodology

### 1. Handling Extreme Imbalance
- **Scale Position Weighting:** In XGBoost, positive class instances are weighted using:
  $$\text{scale\_pos\_weight} = \frac{N_{\text{legitimate}}}{N_{\text{fraud}}}$$
- **Avoidance of Naive Oversampling:** Naive synthetic oversampling on raw datasets alters spatial density in marginal boundaries, increasing False Positives during live inference.

### 2. Model Benchmarking Strategy
- **Supervised Model:** XGBoost Classifier with algorithmic class weighting.
- **Unsupervised Anomaly Detector:** Isolation Forest trained strictly on benign transactions.
- **Primary Metric:** **Precision-Recall AUC (PR-AUC)**. ROC-AUC is rejected due to the severe imbalance ratio inflating True Negative performance.

### 3. Financial Cost Matrix Analysis
Decisions are governed by operational financial costs rather than an arbitrary 0.5 probability cut-off:
$$\text{Total Financial Loss} = C_{\text{FP}} \times \text{FP} + C_{\text{FN}} \times \text{FN}$$
- **Cost of FP ($C_{\text{FP}}$ = $20):** Operational expense of card blocks & customer friction.
- **Cost of FN ($C_{\text{FN}}$ = $250):** Average direct fraudulent chargeback loss + scheme penalties.

### 4. Drift Detection Framework
- **PSI (Population Stability Index):** Monitors feature distribution shifts between training and inference windows.
  $$\text{PSI} = \sum \left( \text{Actual}\% - \text{Expected}\% \right) \times \ln\left(\frac{\text{Actual}\%}{\text{Expected}\%}\right)$$
- **Alert Trigger:** $PSI > 0.25$ triggers an automated alert indicating model degradation.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/your-username/aiml-02-fraud-drift.git](https://github.com/sanj200718/aiml-02-fraud-drift.git)
   cd aiml-02-fraud-drift