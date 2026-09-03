"""
Agent Guardrails and Circuit Breaker System.

Prevents rogue execution, double charging, spamming customers, and unauthorized
actions on high-value corporate transactions.
"""

import hashlib
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Set, Optional, Tuple

from recoverai.data.schemas import (
    PaymentTransaction,
    InterventionType,
    RecoveryDecision,
)
from recoverai.config.settings import settings

logger = logging.getLogger("recoverai.agent.guardrails")


class AgentGuardrailSystem:
    """In-memory and persistent safety validator for agent actions."""

    def __init__(self):
        # Map of idempotency_key -> timestamp
        self._executed_idempotency_keys: Set[str] = set()
        # Map of customer_id -> last_contact_timestamp
        self._customer_last_contact: Dict[str, datetime] = {}
        # Map of transaction_id -> attempt_count
        self._transaction_attempts: Dict[str, int] = {}

    def generate_idempotency_key(self, txn_id: str, action: InterventionType, attempt: int) -> str:
        """Generates a cryptographic idempotency key for an action dispatch."""
        payload = f"{txn_id}:{action.value}:{attempt}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate_action_dispatch(
        self,
        txn: PaymentTransaction,
        decision: RecoveryDecision,
        now: Optional[datetime] = None,
    ) -> Tuple[bool, Optional[str]]:
        """
        Validates whether an action is safe to dispatch.
        
        Returns:
            Tuple of (is_approved: bool, rejection_reason: Optional[str])
        """
        now = now or datetime.now(timezone.utc).replace(tzinfo=None)
        action = decision.recommended_action
        attempt = txn.retry_attempt_count + 1

        # 1. Check Max Attempts Guard
        if txn.retry_attempt_count >= settings.MAX_RECOVERY_ATTEMPTS:
            return False, f"CIRCUIT_BREAKER: Max recovery attempts ({settings.MAX_RECOVERY_ATTEMPTS}) reached for {txn.transaction_id}."

        # 2. Check Idempotency Key (No duplicate execution)
        idem_key = self.generate_idempotency_key(txn.transaction_id, action, attempt)
        if idem_key in self._executed_idempotency_keys:
            return False, f"IDEMPOTENCY_VIOLATION: Duplicate action {action.value} already executed for {txn.transaction_id}."

        # 3. Customer Communication Fatigue Guard
        if action in [InterventionType.WHATSAPP_PAY_LINK, InterventionType.SMS_PAY_LINK]:
            last_contact = self._customer_last_contact.get(txn.customer_id)
            if last_contact is not None:
                elapsed_hours = (now - last_contact).total_seconds() / 3600.0
                if elapsed_hours < settings.COMMUNICATION_COOLDOWN_HOURS:
                    return False, (
                        f"FATIGUE_GUARD: Customer {txn.customer_id} was contacted {elapsed_hours:.1f}h ago. "
                        f"Cooldown of {settings.COMMUNICATION_COOLDOWN_HOURS}h enforced."
                    )

        # 4. High-Value Automated Action Guard
        if txn.amount_inr >= settings.HIGH_VALUE_THRESHOLD_INR and action != InterventionType.MANUAL_ESCALATION:
            return False, (
                f"HIGH_VALUE_CIRCUIT_BREAKER: Transaction of INR {txn.amount_inr:,.2f} requires human sign-off. "
                f"Automated action {action.value} blocked."
            )

        return True, None

    def record_action_executed(
        self,
        txn: PaymentTransaction,
        action: InterventionType,
        now: Optional[datetime] = None,
    ) -> str:
        """Records an action in the guardrail registry upon successful execution."""
        now = now or datetime.now(timezone.utc).replace(tzinfo=None)
        attempt = txn.retry_attempt_count + 1
        idem_key = self.generate_idempotency_key(txn.transaction_id, action, attempt)
        self._executed_idempotency_keys.add(idem_key)
        self._transaction_attempts[txn.transaction_id] = attempt

        if action in [InterventionType.WHATSAPP_PAY_LINK, InterventionType.SMS_PAY_LINK]:
            self._customer_last_contact[txn.customer_id] = now

        return idem_key

    def reset_state(self):
        """Clears in-memory guardrail caches for test isolation."""
        self._executed_idempotency_keys.clear()
        self._customer_last_contact.clear()
        self._transaction_attempts.clear()
