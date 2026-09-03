"""Unit tests for ML models, calibration, and evaluator."""

import pytest
import numpy as np
import pandas as pd

from recoverai.models.recovery_classifier import CalibratedRecoveryClassifier
from recoverai.models.baseline import RuleBasedBaseline, LogisticRegressionBaseline
from recoverai.models.evaluator import ModelEvaluator
from recoverai.features.pipeline import FeaturePipeline


def test_calibrated_recovery_classifier():
    classifier = CalibratedRecoveryClassifier()
    classifier.load()

    feature_pipeline = FeaturePipeline()
    feature_pipeline.load()

    sample_dict = {
        "amount_inr": 2500.0,
        "customer_ltv_inr": 20000.0,
        "hour_of_day": 14,
        "day_of_week": 1,
        "day_of_month": 10,
        "bank_code": "ICICI",
        "customer_tier": "REGULAR",
        "payment_method": "UPI",
        "failure_category": "NETWORK_ERROR",
        "is_recurring": False,
        "retry_attempt_count": 0,
    }

    X_single = feature_pipeline.transform_single(sample_dict)
    prob = classifier.predict_recovery_probability(X_single)[0]

    assert 0.0 <= prob <= 1.0
    assert prob > 0.50  # Network errors on ICICI have strong recovery probability


def test_model_evaluator_metrics():
    y_true = pd.Series([1, 1, 0, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.7, 0.3, 0.1, 0.85, 0.15])
    amounts = pd.Series([1000.0, 2000.0, 500.0, 1500.0, 300.0, 400.0, 2500.0, 600.0])

    eval_out = ModelEvaluator.evaluate_model(
        model_name="TestModel",
        y_true=y_true,
        y_prob=y_prob,
        amounts_inr=amounts,
        threshold=0.5,
    )

    assert eval_out["roc_auc"] == 1.0
    assert eval_out["precision"] == 1.0
    assert eval_out["recall"] == 1.0
    assert eval_out["gross_revenue_recovered_inr"] == 7000.0
    assert eval_out["brier_score"] < 0.10


def test_calibration_impact_case_studies():
    """
    Validates that the Calibration Impact Case Studies displayed in the dashboard
    match actual held-out test transactions and live inference outputs.
    """
    import json
    from recoverai.config.settings import settings
    from recoverai.data.schemas import PaymentTransaction
    from recoverai.decision.engine import ExpectedValueDecisionEngine

    json_path = settings.BASE_DIR / "results" / "calibration_impact_cases.json"
    assert json_path.exists(), "Case studies artifact results/calibration_impact_cases.json must exist."

    with open(json_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    assert len(cases) == 3, "Must contain exactly 3 representative case studies."

    test_path = settings.DATA_DIR / "processed" / "test.csv"
    test_df = pd.read_csv(test_path)
    test_df = test_df.where(pd.notnull(test_df), None)
    txns_by_id = {row["transaction_id"]: PaymentTransaction(**row) for row in test_df.to_dict(orient="records")}

    feature_pipeline = FeaturePipeline()
    feature_pipeline.load()
    classifier = CalibratedRecoveryClassifier()
    classifier.load()
    engine = ExpectedValueDecisionEngine()

    for case in cases:
        txn_id = case["txn_id"]
        assert txn_id in txns_by_id, f"Transaction {txn_id} must exist in held-out test set."
        txn = txns_by_id[txn_id]

        X_single = feature_pipeline.transform_single(txn.model_dump())
        p_cal_live = float(classifier.calibrated_model.predict_proba(X_single)[:, 1][0])
        p_uncal_live = float(classifier.base_model.predict_proba(X_single)[:, 1][0])

        assert np.isclose(case["p_cal"], p_cal_live, atol=1e-4)
        assert np.isclose(case["p_uncal"], p_uncal_live, atol=1e-4)

        dec_live = engine.decide_intervention(txn)
        assert case["act_cal"] == dec_live.recommended_action.value
        assert case["delay_cal"] == dec_live.recommended_delay_hours
        assert np.isclose(case["ev_cal"], dec_live.net_expected_roi_inr, atol=1e-2)

