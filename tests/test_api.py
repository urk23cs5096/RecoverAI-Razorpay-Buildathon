"""Integration tests for FastAPI REST API endpoints."""

import pytest
from fastapi.testclient import TestClient
from recoverai.api.main import app
from recoverai.storage.database import init_db

init_db()
client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["database_connected"] is True


def test_diagnose_api():
    payload = {
        "transaction": {
            "transaction_id": "txn_api_test_01",
            "merchant_id": "merch_saas_metrics",
            "merchant_category": "SaaS_B2B",
            "customer_id": "cust_api_01",
            "customer_tier": "VIP",
            "customer_ltv_inr": 75000.0,
            "preferred_channel": "WHATSAPP",
            "amount_inr": 4999.0,
            "currency": "INR",
            "payment_method": "CARD",
            "bank_code": "HDFC",
            "card_network": "VISA",
            "is_recurring": True,
            "timestamp": "2026-03-01T02:30:00",
            "hour_of_day": 2,
            "day_of_week": 6,
            "gateway_error_code": "U19",
            "gateway_error_description": "NPCI Bank switch down",
            "failure_category": "BANK_DOWNTIME",
            "retry_attempt_count": 0,
        }
    }

    response = client.post("/api/v1/diagnose", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "txn_api_test_01"
    assert "decision" in data
    assert "explanation" in data
    assert data["decision"]["recommended_action"] == "SMART_RETRY_DELAYED"


def test_process_incident_api():
    payload = {
        "transaction": {
            "transaction_id": "txn_api_test_02",
            "merchant_id": "merch_saas_metrics",
            "merchant_category": "SaaS_B2B",
            "customer_id": "cust_api_02",
            "customer_tier": "REGULAR",
            "customer_ltv_inr": 12000.0,
            "preferred_channel": "WHATSAPP",
            "amount_inr": 1499.0,
            "currency": "INR",
            "payment_method": "CARD",
            "bank_code": "ICICI",
            "card_network": "MASTERCARD",
            "is_recurring": False,
            "timestamp": "2026-03-01T15:00:00",
            "hour_of_day": 15,
            "day_of_week": 6,
            "gateway_error_code": "3DS_TIMEOUT",
            "gateway_error_description": "Customer dropped OTP screen",
            "failure_category": "AUTH_FAILED_3DS",
            "retry_attempt_count": 0,
        },
        "simulate_execution": True,
    }

    response = client.post("/api/v1/process-incident", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "txn_api_test_02"
    assert data["action_executed"] is True
    assert data["guardrail_status"] == "APPROVED"
    assert data["final_state"] in ["RECOVERED", "FAILED_TERMINAL", "ACTION_EXECUTED"]


def test_analytics_summary_api():
    response = client.get("/api/v1/analytics/summary")
    assert response.status_code == 200
    data = response.json()
    assert "total_revenue_at_risk_inr" in data
    assert "gross_revenue_recovered_inr" in data


def test_randomized_property_stress_suite():
    """
    Property-based stress test validating the full autonomous pipeline across 250+
    randomized, diverse, boundary-tested transactions.
    """
    import random
    from datetime import datetime, timezone
    from recoverai.data.schemas import (
        PaymentMethod,
        FailureCategory,
        CustomerTier,
        CommunicationChannel,
        InterventionType,
    )

    rng = random.Random(42)

    failure_codes = {
        FailureCategory.BANK_DOWNTIME: ["U19", "91", "SWITCH_DOWN", "BANK_TIMEOUT"],
        FailureCategory.INSUFFICIENT_FUNDS: ["U66", "51", "INSUFFICIENT_FUNDS", "LOW_BALANCE"],
        FailureCategory.AUTH_FAILED_3DS: ["3DS_TIMEOUT", "OTP_EXPIRED", "3DS_FAILED", "AUTH_DROP"],
        FailureCategory.MANDATE_EXPIRED: ["U30", "MANDATE_INACTIVE", "MANDATE_EXPIRED", "TOKEN_REVOKED"],
        FailureCategory.CARD_LIMIT_EXCEEDED: ["61", "LIMIT_EXCEEDED", "DAILY_LIMIT_REACHED"],
        FailureCategory.NETWORK_ERROR: ["NET_ERR", "TIMEOUT_GATEWAY", "CONN_RESET"],
    }

    banks = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "YESB", "PNB", "FEDERAL", "OTHER", "CANARA", "INDUSIND"]
    payment_methods = list(PaymentMethod)
    customer_tiers = list(CustomerTier)
    channels = list(CommunicationChannel)
    card_networks = ["VISA", "MASTERCARD", "RUPAY", "AMEX", "NONE", None]
    merchant_cats = ["SaaS_B2B", "D2C_eCommerce", "EdTech_Subscriptions", "OTT_Media", "B2B_Invoicing"]

    total_cases = 250
    for i in range(total_cases):
        # Explicit boundary testing for early cases
        if i == 0:
            amount = 10.0
        elif i == 1:
            amount = 49999.00
        elif i == 2:
            amount = 49999.99
        elif i == 3:
            amount = 50000.00
        elif i == 4:
            amount = 50000.01
        elif i == 5:
            amount = 50001.00
        elif i == 6:
            amount = 5000000.00  # ₹50 Lakhs
        elif i == 7:
            amount = 4.99  # Micro transaction
        else:
            scale = rng.choice(["micro", "normal", "boundary", "macro"])
            if scale == "micro":
                amount = round(rng.uniform(5.0, 500.0), 2)
            elif scale == "normal":
                amount = round(rng.uniform(500.0, 45000.0), 2)
            elif scale == "boundary":
                amount = round(rng.choice([49990.0, 49999.5, 50000.0, 50000.5, 50010.0]), 2)
            else:
                amount = round(rng.uniform(50000.0, 5000000.0), 2)

        category = rng.choice(list(FailureCategory))
        error_code = rng.choice(failure_codes[category])
        
        # Retry count spread
        if i % 6 == 0:
            retries = 0
        elif i % 6 == 1:
            retries = 1
        elif i % 6 == 2:
            retries = 2
        elif i % 6 == 3:
            retries = 3  # Exact limit
        elif i % 6 == 4:
            retries = 4  # Exceeding limit
        else:
            retries = rng.randint(0, 5)

        hour = rng.randint(0, 23)
        day_of_week = rng.randint(0, 6)
        day_of_month = rng.randint(1, 31)
        customer_ltv = round(rng.choice([0.0, 100.0, 5000.0, 45000.0, 1000000.0]), 2)
        card_net = rng.choice(card_networks)
        mandate_id = f"man_{rng.randint(100, 999)}" if rng.random() > 0.5 else None

        txn_dict = {
            "transaction_id": f"txn_prop_{i:04d}",
            "merchant_id": f"merch_{rng.choice(['alpha', 'beta', 'gamma', 'delta'])}",
            "merchant_category": rng.choice(merchant_cats),
            "customer_id": f"cust_prop_{rng.randint(1, 100):03d}",
            "customer_tier": rng.choice(customer_tiers).value,
            "customer_ltv_inr": customer_ltv,
            "preferred_channel": rng.choice(channels).value,
            "amount_inr": amount,
            "currency": "INR",
            "payment_method": rng.choice(payment_methods).value,
            "bank_code": rng.choice(banks),
            "is_recurring": bool(rng.getrandbits(1)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "hour_of_day": hour,
            "day_of_week": day_of_week,
            "day_of_month": day_of_month,
            "gateway_error_code": error_code,
            "gateway_error_description": f"Random failure description for {error_code}",
            "failure_category": category.value,
            "retry_attempt_count": retries,
        }

        if card_net is not None:
            txn_dict["card_network"] = card_net
        if mandate_id is not None:
            txn_dict["mandate_id"] = mandate_id

        payload = {
            "transaction": txn_dict,
            "simulate_execution": True,
        }

        response = client.post("/api/v1/process-incident", json=payload)
        assert response.status_code == 200, f"Case {i} failed with status {response.status_code}: {response.text}"

        data = response.json()
        assert data["transaction_id"] == f"txn_prop_{i:04d}"
        assert "decision" in data
        assert "explanation" in data
        assert "final_state" in data
        assert "guardrail_status" in data

        decision = data["decision"]
        action = decision["recommended_action"]
        assert action in [e.value for e in InterventionType]

        # Invariant 1: Max retry limit (>= 3 attempts) -> Never execute automated retries or pay links
        if retries >= 3:
            assert action == "NO_ACTION" or not data["action_executed"]

        # Invariant 2: High Value Threshold (>= ₹50,000) -> If retries < 3, action MUST be MANUAL_ESCALATION
        if amount >= 50000.0 and retries < 3:
            assert action == "MANUAL_ESCALATION"

        # Invariant 3: Automated retry on 3DS or Mandate expired -> Strictly forbidden
        if category.value in ["AUTH_FAILED_3DS", "MANDATE_EXPIRED"]:
            assert action not in ["SMART_RETRY_IMMEDIATE", "SMART_RETRY_DELAYED"]

