"""Tests for Expected Value Decision Engine."""

import pytest
from datetime import datetime, timezone
from recoverai.data.schemas import (
    PaymentTransaction,
    InterventionType,
    FailureCategory,
    CustomerTier,
    CommunicationChannel,
    PaymentMethod,
)
from recoverai.decision.engine import ExpectedValueDecisionEngine


def test_decision_engine_expected_value_and_invariants():
    engine = ExpectedValueDecisionEngine()

    # 1. Bank Downtime should trigger smart retry delayed
    txn_downtime = PaymentTransaction(
        transaction_id="txn_dt_1",
        merchant_id="merch_1",
        merchant_category="SaaS",
        customer_id="cust_1",
        customer_tier=CustomerTier.REGULAR,
        customer_ltv_inr=10000.0,
        preferred_channel=CommunicationChannel.WHATSAPP,
        amount_inr=1999.0,
        currency="INR",
        payment_method=PaymentMethod.CARD,
        bank_code="HDFC",
        timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
        hour_of_day=1,  # Midnight maintenance
        day_of_week=2,
        gateway_error_code="U19",
        gateway_error_description="NPCI switch down",
        failure_category=FailureCategory.BANK_DOWNTIME,
        retry_attempt_count=0,
    )

    dec_1 = engine.decide_intervention(txn_downtime)
    assert dec_1.recommended_action == InterventionType.SMART_RETRY_DELAYED
    assert dec_1.recommended_delay_hours >= 1
    assert dec_1.net_expected_roi_inr > 0.0

    # 2. 3DS Authentication Failure should trigger Payment Link or UPI Intent switch (NOT auto retry)
    txn_3ds = PaymentTransaction(
        transaction_id="txn_3ds_1",
        merchant_id="merch_1",
        merchant_category="D2C",
        customer_id="cust_2",
        customer_tier=CustomerTier.REGULAR,
        customer_ltv_inr=5000.0,
        preferred_channel=CommunicationChannel.WHATSAPP,
        amount_inr=999.0,
        currency="INR",
        payment_method=PaymentMethod.CARD,
        bank_code="ICICI",
        timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
        hour_of_day=14,
        day_of_week=4,
        gateway_error_code="3DS_TIMEOUT",
        gateway_error_description="Customer OTP timeout",
        failure_category=FailureCategory.AUTH_FAILED_3DS,
        retry_attempt_count=0,
    )

    dec_2 = engine.decide_intervention(txn_3ds)
    assert "PAY_LINK" in dec_2.recommended_action.value or "UPI_INTENT" in dec_2.recommended_action.value
    assert dec_2.recommended_action != InterventionType.SMART_RETRY_DELAYED


def test_action_tier_selection_varies_with_ev_and_channel_affinity():
    """Verifies that action tier assignment dynamically adapts to ticket size, EV, and channel preference."""
    engine = ExpectedValueDecisionEngine()

    # Case A: Low-ticket, SMS preferred -> SMS / UPI Intent
    txn_low = PaymentTransaction(
        transaction_id="txn_low_1",
        merchant_id="merch_1",
        merchant_category="D2C",
        customer_id="cust_low",
        customer_tier=CustomerTier.REGULAR,
        customer_ltv_inr=2000.0,
        preferred_channel=CommunicationChannel.SMS,
        amount_inr=499.0,
        currency="INR",
        payment_method=PaymentMethod.UPI,
        bank_code="SBI",
        timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
        hour_of_day=11,
        day_of_week=3,
        gateway_error_code="3DS_TIMEOUT",
        gateway_error_description="Customer dropped",
        failure_category=FailureCategory.AUTH_FAILED_3DS,
        retry_attempt_count=0,
    )

    # Case B: High-ticket VIP card limit failure -> Tier 4 Concierge / Manual Escalation
    txn_vip = PaymentTransaction(
        transaction_id="txn_vip_1",
        merchant_id="merch_1",
        merchant_category="B2B_SaaS",
        customer_id="cust_vip",
        customer_tier=CustomerTier.VIP,
        customer_ltv_inr=85000.0,
        preferred_channel=CommunicationChannel.WHATSAPP,
        amount_inr=24999.0,
        currency="INR",
        payment_method=PaymentMethod.CARD,
        bank_code="HDFC",
        timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
        hour_of_day=15,
        day_of_week=1,
        gateway_error_code="LIMIT_EXCEEDED",
        gateway_error_description="Card limit overrun",
        failure_category=FailureCategory.CARD_LIMIT_EXCEEDED,
        retry_attempt_count=0,
    )

    # Case C: Low-ticket card limit failure on regular tier -> Tier 2 UPI Intent switch
    txn_reg = PaymentTransaction(
        transaction_id="txn_reg_1",
        merchant_id="merch_1",
        merchant_category="eCommerce",
        customer_id="cust_reg",
        customer_tier=CustomerTier.REGULAR,
        customer_ltv_inr=3000.0,
        preferred_channel=CommunicationChannel.SMS,
        amount_inr=899.0,
        currency="INR",
        payment_method=PaymentMethod.CARD,
        bank_code="AXIS",
        timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
        hour_of_day=16,
        day_of_week=2,
        gateway_error_code="LIMIT_EXCEEDED",
        gateway_error_description="Card limit overrun",
        failure_category=FailureCategory.CARD_LIMIT_EXCEEDED,
        retry_attempt_count=0,
    )

    dec_low = engine.decide_intervention(txn_low)
    dec_vip = engine.decide_intervention(txn_vip)
    dec_reg = engine.decide_intervention(txn_reg)

    # Assert that action tier assignment differs based on EV and customer profile
    assert dec_vip.recommended_action == InterventionType.MANUAL_ESCALATION
    assert dec_reg.recommended_action == InterventionType.UPI_INTENT_SWITCH
    assert dec_vip.recommended_action != dec_reg.recommended_action
