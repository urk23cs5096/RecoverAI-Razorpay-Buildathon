"""Tests for stochastic recovery simulator and counterfactual benchmark runner."""

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
from recoverai.simulator.engine import RecoverySimulatorEngine
from recoverai.simulator.counterfactual import CounterfactualBenchmarkRunner


def test_recovery_simulator_physics():
    sim = RecoverySimulatorEngine(seed=42)

    # 1. Immediate retry during bank downtime should have very low success rate
    txn_dt = PaymentTransaction(
        transaction_id="txn_dt_sim_1",
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
        hour_of_day=1,
        day_of_week=2,
        gateway_error_code="U19",
        gateway_error_description="Bank downtime",
        failure_category=FailureCategory.BANK_DOWNTIME,
        retry_attempt_count=0,
    )

    out_delayed = sim.simulate_outcome(txn_dt, InterventionType.SMART_RETRY_DELAYED, delay_hours=4)
    assert out_delayed["operational_cost_inr"] > 0
    assert out_delayed["response_channel"] == InterventionType.SMART_RETRY_DELAYED.value

    # 2. No action outcome
    out_none = sim.simulate_outcome(txn_dt, InterventionType.NO_ACTION)
    assert not out_none["is_recovered"]
    assert out_none["amount_recovered_inr"] == 0.0
    assert out_none["operational_cost_inr"] == 0.0


def test_counterfactual_benchmark_runner():
    runner = CounterfactualBenchmarkRunner()

    txn = PaymentTransaction(
        transaction_id="txn_bench_1",
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
        hour_of_day=1,
        day_of_week=2,
        gateway_error_code="U19",
        gateway_error_description="Bank downtime",
        failure_category=FailureCategory.BANK_DOWNTIME,
        retry_attempt_count=0,
    )

    summary = runner.run_benchmark([txn])
    assert "policies" in summary
    assert "RecoverAI" in summary["policies"]
    assert "Rule_Based_Heuristic" in summary["policies"]
    assert "Naive_Immediate_3x" in summary["policies"]
    assert "RecoverAI_Actions_No_ML" in summary["policies"]
    assert summary["total_revenue_at_risk_inr"] == 1999.0


def test_counterfactual_policies_execute_distinct_actions_and_outcomes():
    """
    Regression test ensuring that RecoverAI, Rule-Based Heuristic, and Naive 3x
    execute distinct policies and produce non-identical counterfactual results.
    """
    runner = CounterfactualBenchmarkRunner(simulator_engine=RecoverySimulatorEngine(seed=42))

    categories = [
        FailureCategory.AUTH_FAILED_3DS,
        FailureCategory.BANK_DOWNTIME,
        FailureCategory.NETWORK_ERROR,
        FailureCategory.INSUFFICIENT_FUNDS,
        FailureCategory.MANDATE_EXPIRED,
    ]

    txns = []
    for i in range(50):
        cat = categories[i % len(categories)]
        txns.append(
            PaymentTransaction(
                transaction_id=f"txn_mix_{i}",
                merchant_id="merch_1",
                merchant_category="SaaS_B2B",
                customer_id=f"cust_{i}",
                customer_tier=CustomerTier.REGULAR,
                customer_ltv_inr=12000.0,
                preferred_channel=CommunicationChannel.WHATSAPP,
                amount_inr=2500.0,
                currency="INR",
                payment_method=PaymentMethod.CARD,
                bank_code="HDFC",
                timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                hour_of_day=1 if cat == FailureCategory.BANK_DOWNTIME else 14,
                day_of_week=3,
                day_of_month=28 if cat == FailureCategory.INSUFFICIENT_FUNDS else 15,
                gateway_error_code="U19" if cat == FailureCategory.BANK_DOWNTIME else ("NET_DROP" if cat == FailureCategory.NETWORK_ERROR else "51"),
                gateway_error_description="Failure description",
                failure_category=cat,
                retry_attempt_count=0,
            )
        )

    report = runner.run_benchmark(txns)
    p = report["policies"]

    recov_gross = p["RecoverAI"]["gross_recovered_inr"]
    rule_gross = p["Rule_Based_Heuristic"]["gross_recovered_inr"]
    naive_gross = p["Naive_Immediate_3x"]["gross_recovered_inr"]

    assert recov_gross > rule_gross
    assert rule_gross > naive_gross
    assert p["RecoverAI"]["recovery_rate_pct"] != p["Rule_Based_Heuristic"]["recovery_rate_pct"]
    assert p["Rule_Based_Heuristic"]["recovery_rate_pct"] != p["Naive_Immediate_3x"]["recovery_rate_pct"]


def test_financial_ledger_exact_reconciliation():
    """
    Asserts that for EVERY policy arm in the benchmark:
    Gross Recovered - (Operational Costs + Friction Costs) == Net Revenue Recovered (within 0.01 INR).
    """
    runner = CounterfactualBenchmarkRunner(simulator_engine=RecoverySimulatorEngine(seed=42))

    categories = [
        FailureCategory.AUTH_FAILED_3DS,
        FailureCategory.BANK_DOWNTIME,
        FailureCategory.NETWORK_ERROR,
        FailureCategory.INSUFFICIENT_FUNDS,
        FailureCategory.MANDATE_EXPIRED,
        FailureCategory.CARD_LIMIT_EXCEEDED,
    ]

    txns = []
    for i in range(60):
        cat = categories[i % len(categories)]
        txns.append(
            PaymentTransaction(
                transaction_id=f"txn_ledger_{i}",
                merchant_id="merch_1",
                merchant_category="SaaS_B2B",
                customer_id=f"cust_ledger_{i}",
                customer_tier=CustomerTier.VIP if i % 5 == 0 else CustomerTier.REGULAR,
                customer_ltv_inr=50000.0 if i % 5 == 0 else 8000.0,
                preferred_channel=CommunicationChannel.WHATSAPP if i % 2 == 0 else CommunicationChannel.SMS,
                amount_inr=3500.0 + (i * 150),
                currency="INR",
                payment_method=PaymentMethod.CARD if i % 2 == 0 else PaymentMethod.UPI,
                bank_code="ICICI",
                timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                hour_of_day=1 if cat == FailureCategory.BANK_DOWNTIME else 15,
                day_of_week=2,
                day_of_month=28 if cat == FailureCategory.INSUFFICIENT_FUNDS else 12,
                gateway_error_code="3DS_FAIL",
                gateway_error_description="Failure",
                failure_category=cat,
                retry_attempt_count=0,
            )
        )

    report = runner.run_benchmark(txns)

    for pol_name, p in report["policies"].items():
        gross = p["gross_recovered_inr"]
        ops_cost = p["operational_costs_inr"]
        fric_cost = p["friction_penalties_inr"]
        net = p["net_revenue_recovered_inr"]
        total_deductions = round(ops_cost + fric_cost, 2)
        expected_net = round(gross - total_deductions, 2)

        # Assert exact ledger reconciliation down to 1 paisa (0.01 INR)
        assert abs(net - expected_net) <= 0.01, (
            f"Ledger gap in policy {pol_name}: Gross ({gross}) - Deductions ({total_deductions}) "
            f"= {expected_net} != Reported Net ({net})"
        )
