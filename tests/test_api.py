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
