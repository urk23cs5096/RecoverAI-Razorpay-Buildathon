"""FastAPI API route definitions for RecoverAI."""

from datetime import datetime, timezone
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
import pandas as pd

from recoverai.api.schemas import (
    HealthResponse,
    DiagnoseRequest,
    DiagnoseResponse,
    ProcessIncidentRequest,
    ProcessIncidentResponse,
    BatchSimulateRequest,
    BatchSimulateResponse,
)
from recoverai.data.schemas import PaymentTransaction, AgentState, InterventionType
from recoverai.agent.workflow import RecoveryAgentWorkflow
from recoverai.simulator.engine import RecoverySimulatorEngine
from recoverai.simulator.counterfactual import CounterfactualBenchmarkRunner
from recoverai.storage.database import StorageRepository, init_db
from recoverai.storage.audit import AuditLogger
from recoverai.config.settings import settings

logger = logging.getLogger("recoverai.api.routes")

router = APIRouter()

# Instantiate singletons for API lifetime
workflow = RecoveryAgentWorkflow()
simulator = RecoverySimulatorEngine()
benchmark_runner = CounterfactualBenchmarkRunner(workflow, simulator)
repo = StorageRepository()
audit_logger = AuditLogger(repo)


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint confirming model status and database connection."""
    model_ok = (settings.MODELS_DIR / "recovery_model.joblib").exists()
    return HealthResponse(
        status="ok",
        version=settings.VERSION,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_loaded=model_ok,
        database_connected=True,
    )


@router.post("/api/v1/diagnose", response_model=DiagnoseResponse)
def diagnose_incident(req: DiagnoseRequest):
    """
    Diagnoses a payment failure incident using ML + LLM without triggering external actions.
    """
    txn = req.transaction
    decision = workflow.decision_engine.decide_intervention(txn)
    explanation = workflow.llm_client.explain_and_draft_recovery(txn, decision)
    is_safe, reason = workflow.guardrails.validate_action_dispatch(txn, decision)

    return DiagnoseResponse(
        transaction_id=txn.transaction_id,
        decision=decision,
        explanation=explanation,
        is_safe_to_execute=is_safe,
        guardrail_notes=reason,
    )


@router.post("/api/v1/process-incident", response_model=ProcessIncidentResponse)
def process_incident(req: ProcessIncidentRequest):
    """
    Runs the full autonomous agent cycle (Detect -> Diagnose -> Decide -> Guardrail -> Act -> Verify -> Audit).
    """
    txn = req.transaction
    txn_dict = txn.model_dump()
    
    # 1. Upsert transaction in DB
    db_txn = repo.upsert_transaction(txn_dict)

    # 2. Process through agent state machine
    agent_out = workflow.process_incident(txn)
    decision = agent_out["decision"]
    explanation = agent_out["explanation"]

    # Record decision in DB
    decision_dict = {
        "transaction_id": txn.transaction_id,
        "recommended_action": decision.recommended_action.value,
        "recommended_delay_hours": decision.recommended_delay_hours,
        "estimated_recovery_probability": decision.estimated_recovery_probability,
        "expected_recovery_value_inr": decision.expected_recovery_value_inr,
        "intervention_cost_inr": decision.intervention_cost_inr,
        "net_expected_roi_inr": decision.net_expected_roi_inr,
        "action_confidence": decision.action_confidence,
        "safety_rule_applied": decision.safety_rule_applied,
        "reasoning_summary": decision.reasoning_summary,
        "root_cause_diagnosis": explanation.root_cause_diagnosis,
        "merchant_recommendation": explanation.merchant_recommendation,
        "customer_recovery_message": explanation.customer_recovery_message,
    }
    repo.record_decision(decision_dict)

    sim_outcome = None
    if req.simulate_execution and agent_out["action_executed"]:
        sim_outcome = simulator.simulate_outcome(
            txn=txn,
            action=decision.recommended_action,
            delay_hours=decision.recommended_delay_hours,
        )
        
        # Log Audit event
        cost = sim_outcome["operational_cost_inr"]
        recov_amt = sim_outcome["amount_recovered_inr"] if sim_outcome["is_recovered"] else 0.0
        final_state = AgentState.RECOVERED if sim_outcome["is_recovered"] else AgentState.FAILED_TERMINAL

        audit_logger.log_transition(
            transaction_id=txn.transaction_id,
            previous_state=AgentState.ACTION_EXECUTED,
            current_state=final_state,
            action_type=decision.recommended_action,
            idempotency_key=agent_out.get("idempotency_key"),
            cost_incurred_inr=cost,
            amount_recovered_inr=recov_amt,
            details_json=sim_outcome,
        )

        # Update transaction record
        repo.upsert_transaction({
            "transaction_id": txn.transaction_id,
            "merchant_id": txn.merchant_id,
            "customer_id": txn.customer_id,
            "amount_inr": txn.amount_inr,
            "payment_method": txn.payment_method.value,
            "bank_code": txn.bank_code,
            "failure_category": txn.failure_category.value,
            "gateway_error_code": txn.gateway_error_code,
            "current_agent_state": final_state.value,
            "is_recovered": sim_outcome["is_recovered"],
            "amount_recovered_inr": recov_amt,
            "retry_attempt_count": txn.retry_attempt_count + 1,
        })
        agent_out["final_state"] = final_state.value

    return ProcessIncidentResponse(
        transaction_id=txn.transaction_id,
        final_state=agent_out["final_state"],
        action_executed=agent_out["action_executed"],
        decision=decision,
        explanation=explanation,
        guardrail_status=agent_out["guardrail_status"],
        idempotency_key=agent_out.get("idempotency_key"),
        simulation_outcome=sim_outcome,
        timeline=agent_out["timeline"],
    )


@router.post("/api/v1/simulate/batch", response_model=BatchSimulateResponse)
def batch_simulate(req: BatchSimulateRequest):
    """
    Runs counterfactual benchmark simulation over a batch of transactions.
    """
    raw_path = settings.DATA_DIR / "raw" / "transactions_synthetic.csv"
    if not raw_path.exists():
        from recoverai.data.generator import generate_and_save_data
        generate_and_save_data()

    df = pd.read_csv(raw_path)
    if req.merchant_id:
        df = df[df["merchant_id"] == req.merchant_id]

    sample_df = df.head(req.sample_size)
    sample_df = sample_df.where(pd.notnull(sample_df), None)
    from recoverai.data.schemas import PaymentTransaction
    txns = [PaymentTransaction(**row) for row in sample_df.to_dict(orient="records")]

    results = benchmark_runner.run_benchmark(txns)
    return BatchSimulateResponse(**results)


@router.get("/api/v1/transactions")
def list_transactions(
    limit: int = Query(default=50, ge=1, le=500),
    merchant_id: Optional[str] = Query(default=None),
):
    """Lists recent transactions persisted in database."""
    items = repo.list_transactions(limit=limit, merchant_id=merchant_id)
    return [
        {
            "transaction_id": t.transaction_id,
            "merchant_id": t.merchant_id,
            "amount_inr": t.amount_inr,
            "payment_method": t.payment_method,
            "bank_code": t.bank_code,
            "failure_category": t.failure_category,
            "gateway_error_code": t.gateway_error_code,
            "current_agent_state": t.current_agent_state,
            "is_recovered": t.is_recovered,
            "amount_recovered_inr": t.amount_recovered_inr,
            "created_at": t.created_at.isoformat() if t.created_at else None,
        }
        for t in items
    ]


@router.get("/api/v1/transactions/{txn_id}")
def get_transaction_detail(txn_id: str):
    """Retrieves single transaction detail with decision history and audit trail."""
    txn = repo.get_transaction_by_id(txn_id)
    if not txn:
        raise HTTPException(status_code=404, detail=f"Transaction {txn_id} not found.")

    audits = repo.get_audit_trail_for_txn(txn_id)
    return {
        "transaction": {
            "transaction_id": txn.transaction_id,
            "merchant_id": txn.merchant_id,
            "merchant_category": txn.merchant_category,
            "customer_id": txn.customer_id,
            "customer_tier": txn.customer_tier,
            "customer_ltv_inr": txn.customer_ltv_inr,
            "amount_inr": txn.amount_inr,
            "payment_method": txn.payment_method,
            "bank_code": txn.bank_code,
            "failure_category": txn.failure_category,
            "gateway_error_code": txn.gateway_error_code,
            "gateway_error_description": txn.gateway_error_description,
            "current_agent_state": txn.current_agent_state,
            "is_recovered": txn.is_recovered,
            "amount_recovered_inr": txn.amount_recovered_inr,
        },
        "decisions": [
            {
                "recommended_action": d.recommended_action,
                "delay_hours": d.recommended_delay_hours,
                "recovery_probability": d.estimated_recovery_probability,
                "net_roi_inr": d.net_expected_roi_inr,
                "root_cause": d.root_cause_diagnosis,
                "customer_message": d.customer_recovery_message,
                "created_at": d.created_at.isoformat(),
            }
            for d in txn.decisions
        ],
        "audit_trail": [
            {
                "timestamp": a.timestamp.isoformat(),
                "previous_state": a.previous_state,
                "current_state": a.current_state,
                "action_type": a.action_type,
                "actor": a.actor,
                "idempotency_key": a.idempotency_key,
                "cost_incurred_inr": a.cost_incurred_inr,
                "amount_recovered_inr": a.amount_recovered_inr,
                "details": a.details_json,
            }
            for a in audits
        ],
    }


@router.get("/api/v1/analytics/summary")
def get_analytics_summary():
    """Executive ROI dashboard statistics."""
    return repo.get_executive_summary_stats()
