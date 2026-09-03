"""Unit tests for Data Generator, Validator, and Feature Pipeline."""

import pytest
import pandas as pd
from datetime import datetime

from recoverai.data.generator import SyntheticDataGenerator
from recoverai.data.validator import DataContractValidator
from recoverai.features.extractor import FeatureExtractor
from recoverai.features.pipeline import FeaturePipeline


def test_synthetic_data_generation_and_validation():
    generator = SyntheticDataGenerator(seed=123)
    df = generator.generate_dataset(num_transactions=100)
    
    assert len(df) == 100
    assert "transaction_id" in df.columns
    assert "amount_inr" in df.columns
    assert "failure_category" in df.columns
    assert "recovered_optimal" in df.columns

    # Validate dataset
    cleaned_df, report = DataContractValidator.validate_dataframe(df)
    assert report["is_valid"] is True
    assert report["dropped_rows"] == 0
    assert len(cleaned_df) == 100


def test_data_validator_drops_corrupted_rows():
    corrupted_data = pd.DataFrame([
        {
            "transaction_id": "txn_bad_1",
            "merchant_id": "merch_1",
            "merchant_category": "SaaS",
            "customer_id": "cust_1",
            "customer_tier": "VIP",
            "customer_ltv_inr": 1000.0,
            "preferred_channel": "WHATSAPP",
            "amount_inr": -50.0,  # Negative amount violation
            "currency": "INR",
            "payment_method": "UPI",
            "bank_code": "HDFC",
            "card_network": "NONE",
            "is_recurring": False,
            "timestamp": "2026-01-01 10:00:00",
            "hour_of_day": 10,
            "day_of_week": 3,
            "gateway_error_code": "U19",
            "gateway_error_description": "Bank down",
            "failure_category": "BANK_DOWNTIME",
            "retry_attempt_count": 0,
        },
        {
            "transaction_id": "txn_bad_2",
            "merchant_id": "merch_1",
            "merchant_category": "SaaS",
            "customer_id": "cust_2",
            "customer_tier": "REGULAR",
            "customer_ltv_inr": 1000.0,
            "preferred_channel": "WHATSAPP",
            "amount_inr": 500.0,
            "currency": "INR",
            "payment_method": "INVALID_RAIL",  # Invalid payment method
            "bank_code": "HDFC",
            "card_network": "NONE",
            "is_recurring": False,
            "timestamp": "2026-01-01 10:00:00",
            "hour_of_day": 25,  # Invalid hour
            "day_of_week": 3,
            "gateway_error_code": "U19",
            "gateway_error_description": "Bank down",
            "failure_category": "BANK_DOWNTIME",
            "retry_attempt_count": 0,
        },
    ])

    cleaned_df, report = DataContractValidator.validate_dataframe(corrupted_data)
    assert len(cleaned_df) == 0
    assert report["dropped_rows"] == 2


def test_feature_extraction_and_pipeline():
    sample_row = {
        "amount_inr": 4999.0,
        "customer_ltv_inr": 45000.0,
        "hour_of_day": 1,
        "day_of_week": 2,
        "day_of_month": 29,
        "bank_code": "HDFC",
        "customer_tier": "VIP",
        "payment_method": "CARD",
        "failure_category": "BANK_DOWNTIME",
        "is_recurring": False,
        "retry_attempt_count": 0,
    }

    feats = FeatureExtractor.extract_features_from_dict(sample_row)
    assert "log_amount" in feats
    assert feats["is_maintenance_window"] == 1.0
    assert feats["is_salary_window"] == 1.0
    assert feats["is_card"] == 1.0
    assert feats["is_bank_downtime"] == 1.0
    assert feats["is_transient_failure"] == 1.0
