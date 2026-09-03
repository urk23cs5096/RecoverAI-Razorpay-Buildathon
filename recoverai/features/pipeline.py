"""Feature Pipeline orchestration for training and inference."""

import os
from pathlib import Path
from typing import Tuple, Dict, Any
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from recoverai.features.extractor import FeatureExtractor
from recoverai.config.settings import settings


class FeaturePipeline:
    """Manages feature extraction, scaling, persistence, and inference preprocessing."""

    def __init__(self, scaler_path: Path = None):
        self.scaler_path = scaler_path or (settings.MODELS_DIR / "scaler.joblib")
        self.scaler: StandardScaler = StandardScaler()
        self.feature_columns: list[str] = []
        self.is_fitted: bool = False

    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Extracts features and fits the scaler strictly on the training set (zero leakage).
        """
        X_raw = FeatureExtractor.transform_dataframe(df)
        self.feature_columns = list(X_raw.columns)
        
        X_scaled_array = self.scaler.fit_transform(X_raw)
        X_scaled = pd.DataFrame(X_scaled_array, columns=self.feature_columns, index=df.index)
        
        # Target label
        y = df["recovered_optimal"].astype(int)
        self.is_fitted = True
        self.save()
        return X_scaled, y

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms validation or test DataFrame using the fitted scaler."""
        if not self.is_fitted:
            self.load()
            
        X_raw = FeatureExtractor.transform_dataframe(df)
        # Ensure column order matches
        X_raw = X_raw[self.feature_columns]
        X_scaled_array = self.scaler.transform(X_raw)
        return pd.DataFrame(X_scaled_array, columns=self.feature_columns, index=df.index)

    def transform_single(self, record_dict: Dict[str, Any]) -> pd.DataFrame:
        """Prepares a single transaction dictionary for model inference."""
        if not self.is_fitted:
            self.load()
        feat_dict = FeatureExtractor.extract_features_from_dict(record_dict)
        raw_df = pd.DataFrame([feat_dict])[self.feature_columns]
        scaled_array = self.scaler.transform(raw_df)
        return pd.DataFrame(scaled_array, columns=self.feature_columns)

    def save(self) -> None:
        """Saves fitted scaler and feature column metadata."""
        settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"scaler": self.scaler, "feature_columns": self.feature_columns},
            self.scaler_path,
        )

    def load(self) -> None:
        """Loads fitted scaler from disk."""
        if not self.scaler_path.exists():
            raise FileNotFoundError(
                f"Feature scaler artifact not found at {self.scaler_path}. Please train model first."
            )
        data = joblib.load(self.scaler_path)
        self.scaler = data["scaler"]
        self.feature_columns = data["feature_columns"]
        self.is_fitted = True
