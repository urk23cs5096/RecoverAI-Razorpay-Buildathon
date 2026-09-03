"""
Calibrated ML Model for Payment Recovery Probability Estimation.

Utilizes Gradient Boosted Decision Trees (LightGBM / HistGradientBoosting) with 
CalibratedClassifierCV (Platt Scaling) to generate reliable, mathematically sound probabilities
essential for financial Expected Value decisioning.
"""

import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from recoverai.config.settings import settings

logger = logging.getLogger("recoverai.models.recovery_classifier")


class CalibratedRecoveryClassifier:
    """
    Trained Gradient Boosted Classifier with Probability Calibration for accurate ROI forecasting.
    """

    def __init__(self, random_state: int = 42, model_path: Optional[Path] = None):
        self.random_state = random_state
        self.model_path = model_path or (settings.MODELS_DIR / "recovery_model.joblib")
        self.base_model = None
        self.calibrated_model: Optional[CalibratedClassifierCV] = None
        self.feature_names: list[str] = []
        self.is_fitted: bool = False

    def _build_base_estimator(self):
        """Constructs the base tree ensemble with regularized hyperparameters."""
        if HAS_LIGHTGBM:
            logger.info("Using LightGBM as base estimator.")
            return lgb.LGBMClassifier(
                n_estimators=150,
                learning_rate=0.03,
                num_leaves=15,
                max_depth=4,
                min_child_samples=30,
                subsample=0.85,
                colsample_bytree=0.80,
                random_state=self.random_state,
                importance_type="gain",
                verbose=-1,
            )
        else:
            logger.info("LightGBM not detected; using scikit-learn HistGradientBoostingClassifier.")
            return HistGradientBoostingClassifier(
                max_iter=150,
                learning_rate=0.03,
                max_leaf_nodes=15,
                max_depth=4,
                min_samples_leaf=30,
                random_state=self.random_state,
            )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame] = None, y_val: Optional[pd.Series] = None):
        """
        Fits base tree model and calibrates probabilities using 5-fold cross-validation
        with Platt Scaling.
        """
        self.feature_names = list(X_train.columns)
        self.base_model = self._build_base_estimator()

        logger.info(f"Fitting base model on {len(X_train)} samples with {len(self.feature_names)} features...")
        self.base_model.fit(X_train, y_train)

        # Calibrate using Platt Scaling (sigmoid) with 5-fold cross-validation
        logger.info("Calibrating model probabilities using CalibratedClassifierCV (Platt Scaling)...")
        self.calibrated_model = CalibratedClassifierCV(
            estimator=self.base_model,
            method="sigmoid",
            cv=5,
        )
        self.calibrated_model.fit(X_train, y_train)
        self.is_fitted = True

        self.save()
        logger.info("Model fitting and calibration complete.")
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Returns calibrated 2D class probabilities [P(0), P(1)]."""
        if not self.is_fitted:
            self.load()
        return self.calibrated_model.predict_proba(X)

    def predict_recovery_probability(self, X: pd.DataFrame) -> np.ndarray:
        """Returns 1D array of recovery probabilities P(Recovered=1)."""
        return self.predict_proba(X)[:, 1]

    def predict(self, X: pd.DataFrame, threshold: float = 0.50) -> np.ndarray:
        """Classifies recovery based on probability threshold."""
        probs = self.predict_recovery_probability(X)
        return (probs >= threshold).astype(int)

    def get_feature_importances(self) -> Dict[str, float]:
        """Extracts normalized feature importance dictionary."""
        if not self.is_fitted:
            self.load()
        if hasattr(self.base_model, "feature_importances_"):
            importances = self.base_model.feature_importances_
            total = sum(importances) if sum(importances) > 0 else 1.0
            norm_importances = [float(x) / total for x in importances]
            return dict(sorted(zip(self.feature_names, norm_importances), key=lambda x: x[1], reverse=True))
        return {feat: 1.0 / len(self.feature_names) for feat in self.feature_names}

    def save(self) -> None:
        """Persists trained model artifact to disk."""
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "calibrated_model": self.calibrated_model,
                "base_model": self.base_model,
                "feature_names": self.feature_names,
            },
            self.model_path,
        )
        logger.info(f"Saved model to {self.model_path}")

    def load(self) -> None:
        """Loads model artifact from disk."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found at {self.model_path}. Train model first.")
        data = joblib.load(self.model_path)
        self.calibrated_model = data["calibrated_model"]
        self.base_model = data["base_model"]
        self.feature_names = data["feature_names"]
        self.is_fitted = True
