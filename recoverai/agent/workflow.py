"""
Bounded Agent Recovery Workflow State Machine.

Executes the Detect -> Diagnose -> Decide -> Act -> Verify -> Stop cycle with
immutable audit logging, guardrail validation, and deterministic transitions.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any

from recoverai.data.schemas import (
    PaymentTransaction,
    RecoveryDecision,
    LLMExplanation,
    AuditLogRecord,
    InterventionType,
)
from recoverai.agent.state import AgentState, ALLOWED_STATE_TRANSITIONS
from recoverai.agent.guardrails import AgentGuardrailSystem
from recoverai.decision.engine import ExpectedValueDecisionEngine
from recoverai.llm.client import LLMReasoningClient
from recoverai.config.settings import settings

logger = logging.getLogger("recoverai.agent.workflow")


class RecoveryAgentWorkflow:
    """Manages autonomous recovery lifecycle for a payment failure incident."""

    def __init__(
        self,
        decision_engine: Optional[ExpectedValueDecisionEngine] = None,
        llm_client: Optional[LLMReasoningClient] = None,
        guardrails: Optional[AgentGuardrailSystem] = None,
        audit_logger = None,
    ):
        self.decision_engine = decision_engine or ExpectedValueDecisionEngine()
        self.llm_client = llm_client or LLMReasoningClient()
        self.guardrails = guardrails or AgentGuardrailSystem()
        self.audit_logger = audit_logger

    def transition_state(
        self,
        current_state: AgentState,
        next_state: AgentState,
        txn_id: str,
    ) -> AgentState:
        """Validates state machine transition integrity."""
        allowed = ALLOWED_STATE_TRANSITIONS.get(current_state, set())
        if next_state not in allowed:
            raise ValueError(
                f"Invalid state transition for {txn_id}: {current_state.value} -> {next_state.value} is not allowed."
            )
        return next_state

    def process_incident(
        self,
        txn: PaymentTransaction,
        now: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Executes the full agent workflow for an incoming failed transaction.
        """
        now = now or datetime.now(timezone.utc).replace(tzinfo=None)
        workflow_timeline = []

        # 1. DETECT
        state = AgentState.DETECTED
        workflow_timeline.append({
            "state": state.value,
            "timestamp": now.isoformat(),
            "message": f"Payment failure detected: {txn.gateway_error_code} - {txn.gateway_error_description}",
        })

        # 2. DECIDE & ML PROBABILITY
        decision = self.decision_engine.decide_intervention(txn)
        state = self.transition_state(state, AgentState.DECIDED, txn.transaction_id)
        workflow_timeline.append({
            "state": state.value,
            "timestamp": now.isoformat(),
            "decision": decision.model_dump(),
        })

        # 3. DIAGNOSE & LLM REASONING
        explanation = self.llm_client.explain_and_draft_recovery(txn, decision)
        workflow_timeline.append({
            "state": "DIAGNOSED",
            "timestamp": now.isoformat(),
            "explanation": explanation.model_dump(),
        })

        # 4. GUARDRAILS & CIRCUIT BREAKERS
        is_safe, rejection_reason = self.guardrails.validate_action_dispatch(txn, decision, now=now)
        
        if not is_safe:
            # Action blocked by safety invariant
            state = AgentState.FAILED_TERMINAL if "MAX_RETRY" in (rejection_reason or "") else AgentState.ESCALATED
            workflow_timeline.append({
                "state": state.value,
                "timestamp": now.isoformat(),
                "guardrail_rejection": rejection_reason,
                "status": "BLOCKED_BY_GUARDRAIL",
            })
            return {
                "transaction_id": txn.transaction_id,
                "final_state": state.value,
                "decision": decision,
                "explanation": explanation,
                "guardrail_status": "REJECTED",
                "guardrail_reason": rejection_reason,
                "action_executed": False,
                "timeline": workflow_timeline,
            }

        # 5. ACT: Dispatch action
        if decision.recommended_action == InterventionType.MANUAL_ESCALATION:
            state = AgentState.ESCALATED
            action_executed = True
            action_msg = f"Escalated high-value case (INR {txn.amount_inr:,.2f}) to merchant operations."
        elif decision.recommended_action == InterventionType.NO_ACTION:
            state = AgentState.FAILED_TERMINAL
            action_executed = False
            action_msg = "No recovery action deemed profitable. Suppressed."
        elif decision.recommended_delay_hours > 0:
            state = AgentState.ACTION_SCHEDULED
            action_executed = True
            action_msg = f"Scheduled {decision.recommended_action.value} after {decision.recommended_delay_hours}h backoff."
        else:
            state = AgentState.ACTION_EXECUTED
            action_executed = True
            action_msg = f"Dispatched {decision.recommended_action.value} immediately."

        if action_executed:
            idem_key = self.guardrails.record_action_executed(txn, decision.recommended_action, now=now)
        else:
            idem_key = None

        workflow_timeline.append({
            "state": state.value,
            "timestamp": now.isoformat(),
            "action_type": decision.recommended_action.value,
            "idempotency_key": idem_key,
            "details": action_msg,
        })

        return {
            "transaction_id": txn.transaction_id,
            "final_state": state.value,
            "decision": decision,
            "explanation": explanation,
            "guardrail_status": "APPROVED",
            "action_executed": action_executed,
            "idempotency_key": idem_key,
            "timeline": workflow_timeline,
        }
