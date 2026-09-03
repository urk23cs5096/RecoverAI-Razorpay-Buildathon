"""Feature extraction and engineering for RecoverAI payment failure analysis."""

import math
import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from recoverai.data.schemas import (
    PaymentTransaction,
    FailureCategory,
    CustomerTier,
    PaymentMethod,
)

logger = logging.getLogger("recoverai.features.extractor")

BANK_RELIABILITY_MAP = {
    "HDFC": 0.94,
    "ICICI": 0.93,
    "KOTAK": 0.91,
    "AXIS": 0.89,
    "SBI": 0.84,
    "YESB": 0.82,
    "PNB": 0.78,
    "FEDERAL": 0.80,
    "OTHER": 0.75,
}

CUSTOMER_TIER_SCORE_MAP = {
    CustomerTier.VIP.value: 3.0,
    CustomerTier.REGULAR.value: 2.0,
    CustomerTier.NEW.value: 1.0,
    CustomerTier.AT_RISK.value: 0.0,
}


class FeatureExtractor:
    """Extracts engineered ML features from raw transaction records with zero temporal leakage."""

    @staticmethod
    def extract_features_from_dict(row: Dict[str, Any]) -> Dict[str, float]:
        """Extracts numerical & encoded features from a single transaction dictionary."""
        amount = float(row.get("amount_inr", 0.0))
        ltv = float(row.get("customer_ltv_inr", 0.0))
        hour = int(row.get("hour_of_day", 12))
        day_of_week = int(row.get("day_of_week", 0))
        day_of_month = int(row.get("day_of_month", 15))
        bank_code = str(row.get("bank_code", "OTHER")).upper()
        tier = str(row.get("customer_tier", CustomerTier.REGULAR.value))
        method = str(row.get("payment_method", PaymentMethod.UPI.value))
        category = str(row.get("failure_category", FailureCategory.NETWORK_ERROR.value))
        is_recurring = int(bool(row.get("is_recurring", False)))
        retry_count = int(row.get("retry_attempt_count", 0))

        # Temporal indicators
        is_maintenance_window = 1.0 if hour in [23, 0, 1, 2, 3] else 0.0
        is_salary_window = 1.0 if (day_of_month >= 28 or day_of_month <= 5) else 0.0
        
        # Cyclical temporal encodings
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)
        day_sin = math.sin(2 * math.pi * day_of_week / 7.0)
        day_cos = math.cos(2 * math.pi * day_of_week / 7.0)

        # Financial & customer value ratios
        log_amount = math.log1p(max(0.0, amount))
        log_ltv = math.log1p(max(0.0, ltv))
        amount_to_ltv_ratio = amount / (ltv + 1.0)
        tier_score = CUSTOMER_TIER_SCORE_MAP.get(tier, 2.0)
        bank_reliability = BANK_RELIABILITY_MAP.get(bank_code, 0.78)

        # Payment Rail Indicators
        is_card = 1.0 if "CARD" in method else 0.0
        is_upi = 1.0 if "UPI" in method else 0.0
        is_netbanking = 1.0 if "NETBANKING" in method else 0.0

        # Failure Category Flags
        is_bank_downtime = 1.0 if category == FailureCategory.BANK_DOWNTIME.value else 0.0
        is_insufficient_funds = 1.0 if category == FailureCategory.INSUFFICIENT_FUNDS.value else 0.0
        is_3ds_auth_failed = 1.0 if category == FailureCategory.AUTH_FAILED_3DS.value else 0.0
        is_mandate_expired = 1.0 if category == FailureCategory.MANDATE_EXPIRED.value else 0.0
        is_card_limit_exceeded = 1.0 if category == FailureCategory.CARD_LIMIT_EXCEEDED.value else 0.0
        is_network_error = 1.0 if category == FailureCategory.NETWORK_ERROR.value else 0.0

        # High-order interaction signals
        is_transient_failure = 1.0 if (is_bank_downtime or is_network_error) else 0.0
        is_user_action_required = 1.0 if (is_3ds_auth_failed or is_mandate_expired or is_insufficient_funds) else 0.0

        return {
            "log_amount": log_amount,
            "log_ltv": log_ltv,
            "amount_to_ltv_ratio": amount_to_ltv_ratio,
            "tier_score": tier_score,
            "bank_reliability": bank_reliability,
            "is_maintenance_window": is_maintenance_window,
            "is_salary_window": is_salary_window,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "day_sin": day_sin,
            "day_cos": day_cos,
            "is_recurring": float(is_recurring),
            "retry_attempt_count": float(retry_count),
            "is_card": is_card,
            "is_upi": is_upi,
            "is_netbanking": is_netbanking,
            "is_bank_downtime": is_bank_downtime,
            "is_insufficient_funds": is_insufficient_funds,
            "is_3ds_auth_failed": is_3ds_auth_failed,
            "is_mandate_expired": is_mandate_expired,
            "is_card_limit_exceeded": is_card_limit_exceeded,
            "is_network_error": is_network_error,
            "is_transient_failure": is_transient_failure,
            "is_user_action_required": is_user_action_required,
        }

    @classmethod
    def transform_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms a full DataFrame into feature matrix DataFrame."""
        features_list = [
            cls.extract_features_from_dict(row)
            for row in df.to_dict(orient="records")
        ]
        feature_df = pd.DataFrame(features_list, index=df.index)
        return feature_df
