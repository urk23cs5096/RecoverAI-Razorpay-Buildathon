"""Tests for Agent Guardrails, Safety Circuit Breakers, and State Transitions."""

from datetime import datetime, timezone, timedelta
import pytest

from recoverai.data.schemas import (
    PaymentTransaction,
    InterventionType,
    FailureCategory,
    CustomerTier,
    CommunicationChannel,
    PaymentMethod,
    RecoveryDecision,
)
from recoverai.agent.guardrails import AgentGuardrailSystem
from recoverai.agent.workflow import RecoveryAgentWorkflow
from recoverai.config.settings import settings


def _create_mock_txn(
    amount: float = 1500.0,
    retries: int = 0,
    category: FailureCategory = FailureCategory.BANK_DOWNTIME,
) -> PaymentTransaction:
    return PaymentTransaction(
        transaction_id="txn_test_101",
        merchant_id="merch_test",
        merchant_category="SaaS_B2B",
        customer_id="cust_test_01",
        customer_tier=CustomerTier.REGULAR,
        customer_ltv_inr=15000.0,
        preferred_channel=CommunicationChannel.WHATSAPP,
        amount_inr=amount,
        currency="INR",
        payment_method=PaymentMethod.CARD,
        bank_code="HDFC",
        card_network="VISA",
        is_recurring=False,
        timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
        hour_of_day=2,
        day_of_week=3,
        is_failed=True,
        gateway_error_code="U19",
        gateway_error_description="NPCI server down",
        failure_category=category,
        retry_attempt_count=retries,
    )


def test_guardrail_max_retries_limit():
    guardrails = AgentGuardrailSystem()
    txn = _create_mock_txn(retries=3)  # Already reached max 3 retries
    decision = RecoveryDecision(
        transaction_id=txn.transaction_id,
        recommended_action=InterventionType.SMART_RETRY_DELAYED,
        recommended_delay_hours=2,
        estimated_recovery_probability=0.8,
        expected_recovery_value_inr=1200.0,
        intervention_cost_inr=0.5,
        net_expected_roi_inr=1199.5,
        action_confidence=0.8,
        reasoning_summary="Test",
    )

    is_safe, reason = guardrails.validate_action_dispatch(txn, decision)
    assert is_safe is False
    assert "CIRCUIT_BREAKER" in reason


def test_guardrail_high_value_escalation():
    guardrails = AgentGuardrailSystem()
    txn = _create_mock_txn(amount=75000.0)  # > 50,000 threshold
    decision = RecoveryDecision(
        transaction_id=txn.transaction_id,
        recommended_action=InterventionType.WHATSAPP_PAY_LINK,  # Auto link blocked
        recommended_delay_hours=0,
        estimated_recovery_probability=0.8,
        expected_recovery_value_inr=60000.0,
        intervention_cost_inr=0.8,
        net_expected_roi_inr=59999.2,
        action_confidence=0.8,
        reasoning_summary="Test",
    )

    is_safe, reason = guardrails.validate_action_dispatch(txn, decision)
    assert is_safe is False
    assert "HIGH_VALUE_CIRCUIT_BREAKER" in reason


def test_guardrail_customer_fatigue_cooldown():
    guardrails = AgentGuardrailSystem()
    txn = _create_mock_txn()
    decision = RecoveryDecision(
        transaction_id=txn.transaction_id,
        recommended_action=InterventionType.WHATSAPP_PAY_LINK,
        recommended_delay_hours=0,
        estimated_recovery_probability=0.8,
        expected_recovery_value_inr=1200.0,
        intervention_cost_inr=0.8,
        net_expected_roi_inr=1199.2,
        action_confidence=0.8,
        reasoning_summary="Test",
    )

    now = datetime.now(timezone.utc).replace(tzinfo=None)
    # 1. First dispatch is safe
    is_safe_1, _ = guardrails.validate_action_dispatch(txn, decision, now=now)
    assert is_safe_1 is True
    guardrails.record_action_executed(txn, decision.recommended_action, now=now)

    # 2. Second dispatch 2 hours later to same customer for a distinct transaction must be blocked by fatigue guard
    txn_2 = _create_mock_txn()
    txn_2.transaction_id = "txn_test_102"
    decision_2 = decision.model_copy(update={"transaction_id": "txn_test_102"})
    is_safe_2, reason_2 = guardrails.validate_action_dispatch(txn_2, decision_2, now=now + timedelta(hours=2))
    assert is_safe_2 is False
    assert "FATIGUE_GUARD" in reason_2


def test_workflow_end_to_end():
    workflow = RecoveryAgentWorkflow()
    txn = _create_mock_txn()
    result = workflow.process_incident(txn)

    assert result["transaction_id"] == "txn_test_101"
    assert result["guardrail_status"] == "APPROVED"
    assert "decision" in result
    assert "explanation" in result
    assert len(result["timeline"]) > 0


def test_run_demo_cli_execution_and_audit_provenance():
    """
    Verifies that run_demo.py executes the end-to-end pipeline cleanly on txn_102116
    and records an immutable audit log with full provenance in the database.
    """
    from run_demo import run_cli_demo
    from recoverai.storage.database import StorageRepository

    demo_result = run_cli_demo(target_txn_id="txn_102116")

    assert demo_result["transaction_id"] == "txn_102116"
    assert demo_result["is_recovered"] is True
    assert demo_result["amount_recovered_inr"] == 119334.92
    assert demo_result["p_cal"] > 0.60
    assert demo_result["ev_delta"] > 0.0

    repo = StorageRepository()
    audits = repo.get_audit_trail_for_txn("txn_102116")
    assert len(audits) >= 2, "Must log at least 2 lifecycle state transitions."

    action_event = next((a for a in audits if a.action_type == "MANUAL_ESCALATION"), None)
    assert action_event is not None
    assert action_event.idempotency_key is not None
    assert len(action_event.idempotency_key) == 64  # Valid SHA-256 hash
    assert action_event.details_json is not None
    assert "calibrated_probability" in action_event.details_json
    assert "raw_uncalibrated_probability" in action_event.details_json
    assert "guardrails_passed" in action_event.details_json

    db_txn = repo.get_transaction_by_id("txn_102116")
    assert db_txn is not None
    assert db_txn.is_recovered is True
    assert db_txn.amount_recovered_inr == 119334.92

