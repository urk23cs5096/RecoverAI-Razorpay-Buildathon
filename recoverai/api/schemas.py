"""API Request and Response schemas."""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field

from recoverai.data.schemas import (
    PaymentTransaction,
    RecoveryDecision,
    LLMExplanation,
    AgentState,
    InterventionType,
)


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str = "1.0.0"
    timestamp: str
    model_loaded: bool
    database_connected: bool


class DiagnoseRequest(BaseModel):
    transaction: PaymentTransaction


class DiagnoseResponse(BaseModel):
    transaction_id: str
    decision: RecoveryDecision
    explanation: LLMExplanation
    is_safe_to_execute: bool
    guardrail_notes: Optional[str] = None


class ProcessIncidentRequest(BaseModel):
    transaction: PaymentTransaction
    simulate_execution: bool = True


class ProcessIncidentResponse(BaseModel):
    transaction_id: str
    final_state: str
    action_executed: bool
    decision: RecoveryDecision
    explanation: LLMExplanation
    guardrail_status: str
    idempotency_key: Optional[str] = None
    simulation_outcome: Optional[Dict[str, Any]] = None
    timeline: List[Dict[str, Any]]


class BatchSimulateRequest(BaseModel):
    sample_size: int = Field(default=500, ge=10, le=5000)
    merchant_id: Optional[str] = None


class BatchSimulateResponse(BaseModel):
    total_transactions: int
    total_revenue_at_risk_inr: float
    policies: Dict[str, Any]
    comparison: Dict[str, Any]
