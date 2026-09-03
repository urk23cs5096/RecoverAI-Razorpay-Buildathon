"""Data validation and contract verification layer for RecoverAI."""

import logging
from typing import Tuple, List, Dict, Any
import pandas as pd
from pydantic import ValidationError

from recoverai.data.schemas import (
    PaymentTransaction,
    PaymentMethod,
    FailureCategory,
    CustomerTier,
)

logger = logging.getLogger("recoverai.data.validator")


class DataContractValidator:
    """Validates raw incoming transaction datasets and streaming payloads against contracts."""

    REQUIRED_COLUMNS = [
        "transaction_id",
        "merchant_id",
        "merchant_category",
        "customer_id",
        "customer_tier",
        "customer_ltv_inr",
        "preferred_channel",
        "amount_inr",
        "currency",
        "payment_method",
        "bank_code",
        "card_network",
        "is_recurring",
        "timestamp",
        "hour_of_day",
        "day_of_week",
        "gateway_error_code",
        "gateway_error_description",
        "failure_category",
        "retry_attempt_count",
    ]

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validates a full DataFrame for schema compliance, missing values, and domain rules.
        
        Returns:
            Tuple of (cleaned_valid_df, validation_report_dict)
        """
        initial_count = len(df)
        report: Dict[str, Any] = {
            "initial_rows": initial_count,
            "missing_column_errors": [],
            "schema_errors": 0,
            "domain_violations": 0,
            "clean_rows": 0,
            "dropped_rows": 0,
            "is_valid": True,
        }

        # 1. Check required columns
        missing_cols = [col for col in cls.REQUIRED_COLUMNS if col not in df.columns]
        if missing_cols:
            report["missing_column_errors"] = missing_cols
            report["is_valid"] = False
            logger.error(f"Missing required columns in dataset: {missing_cols}")
            raise ValueError(f"Dataset missing required columns: {missing_cols}")

        valid_mask = pd.Series(True, index=df.index)

        # 2. Domain rules
        # Amounts must be strictly positive
        amount_invalid = (df["amount_inr"] <= 0) | (df["amount_inr"].isna())
        valid_mask &= ~amount_invalid
        report["domain_violations"] += int(amount_invalid.sum())

        # Retry count must be non-negative
        retry_invalid = (df["retry_attempt_count"] < 0) | (df["retry_attempt_count"].isna())
        valid_mask &= ~retry_invalid
        report["domain_violations"] += int(retry_invalid.sum())

        # Hours must be 0-23, Days 0-6
        hour_invalid = (df["hour_of_day"] < 0) | (df["hour_of_day"] > 23)
        day_invalid = (df["day_of_week"] < 0) | (df["day_of_week"] > 6)
        valid_mask &= ~(hour_invalid | day_invalid)

        # Payment methods & failure categories in valid enums
        valid_methods = set(m.value for m in PaymentMethod)
        method_invalid = ~df["payment_method"].isin(valid_methods)
        valid_mask &= ~method_invalid

        valid_categories = set(c.value for c in FailureCategory)
        category_invalid = ~df["failure_category"].isin(valid_categories)
        valid_mask &= ~category_invalid

        cleaned_df = df[valid_mask].copy()
        report["clean_rows"] = len(cleaned_df)
        report["dropped_rows"] = initial_count - len(cleaned_df)

        if report["dropped_rows"] > 0:
            logger.warning(
                f"DataContractValidator dropped {report['dropped_rows']}/{initial_count} invalid records."
            )

        return cleaned_df, report

    @classmethod
    def validate_single_event(cls, payload: Dict[str, Any]) -> PaymentTransaction:
        """Validates a single transaction payload using Pydantic."""
        try:
            return PaymentTransaction(**payload)
        except ValidationError as e:
            logger.error(f"Schema validation error on single transaction: {e}")
            raise e
