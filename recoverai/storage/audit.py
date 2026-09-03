"""Audit logger interface for recording state transitions and actions."""

import logging
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from recoverai.storage.database import StorageRepository
from recoverai.data.schemas import AgentState, InterventionType

logger = logging.getLogger("recoverai.storage.audit")


class AuditLogger:
    """Provides high-level helpers for audit event logging."""

    def __init__(self, repo: Optional[StorageRepository] = None):
        self.repo = repo or StorageRepository()

    def log_transition(
        self,
        transaction_id: str,
        previous_state: AgentState,
        current_state: AgentState,
        action_type: InterventionType,
        idempotency_key: Optional[str] = None,
        cost_incurred_inr: float = 0.0,
        amount_recovered_inr: float = 0.0,
        details_json: Optional[Dict[str, Any]] = None,
    ):
        """Records an immutable audit event."""
        event_dict = {
            "transaction_id": transaction_id,
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None),
            "previous_state": previous_state.value,
            "current_state": current_state.value,
            "action_type": action_type.value,
            "actor": "RecoverAI_Agent",
            "idempotency_key": idempotency_key,
            "cost_incurred_inr": cost_incurred_inr,
            "amount_recovered_inr": amount_recovered_inr,
            "details_json": details_json or {},
        }
        rec = self.repo.record_audit_event(event_dict)
        logger.info(
            f"Audit recorded: [{transaction_id}] {previous_state.value} -> {current_state.value} | "
            f"Action: {action_type.value} | Cost: INR {cost_incurred_inr:.2f} | Recovered: INR {amount_recovered_inr:.2f}"
        )
        return rec
