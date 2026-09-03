"""Baseline models for payment recovery benchmark comparison."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from typing import Dict, Any, Tuple


class RuleBasedBaseline:
    """
    Industry-standard rule-based recovery heuristic:
    - Retries BANK_DOWNTIME and NETWORK_ERROR blindly.
    - Sends SMS for INSUFFICIENT_FUNDS.
    - Drops 3DS and MANDATE failures.
    """

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Heuristic probability score properly handling both scaled and unscaled features."""
        probs = []
        for _, row in X.iterrows():
            # In standardized features (StandardScaler), positive value indicates feature is present (1)
            is_downtime = float(row.get("is_bank_downtime", 0)) > 0.0
            is_net_err = float(row.get("is_network_error", 0)) > 0.0
            is_insufficient = float(row.get("is_insufficient_funds", 0)) > 0.0

            if is_downtime or is_net_err:
                p = 0.50
            elif is_insufficient:
                p = 0.30
            else:
                p = 0.10
            probs.append([1.0 - p, p])
        return np.array(probs)

    def predict(self, X: pd.DataFrame, threshold: float = 0.40) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


class LogisticRegressionBaseline:
    """Standard uncalibrated Logistic Regression baseline."""

    def __init__(self, random_state: int = 42):
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        )

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)
