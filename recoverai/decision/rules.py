"""
Hard Business Rules and Invariants for Payment Recovery.

These deterministic rules enforce non-negotiable safety guardrails and domain constraints.
"""

import logging
from datetime import datetime, timezone
from typing import Optional, Tuple, Set
from recoverai.data.schemas import (
    PaymentTransaction,
    InterventionType,
    FailureCategory,
)
from recoverai.config.settings import settings

logger = logging.getLogger("recoverai.decision.rules")


class BusinessRuleEngine:
    """Enforces strict, non-negotiable safety guardrails on recovery decisions."""

    @classmethod
    def apply_hard_rules(
        cls,
        txn: PaymentTransaction,
        now: Optional[datetime] = None,
    ) -> Tuple[Optional[InterventionType], Optional[str], int]:
        """
        Evaluates hard deterministic invariants.
        
        Returns:
            Tuple of (forced_action, rule_reason, forced_delay_hours)
            If no hard rule overrides, returns (None, None, 0).
        """
        now = now or datetime.now(timezone.utc).replace(tzinfo=None)

        # Invariant 1: Max Retry Limit (Never retry infinitely)
        if txn.retry_attempt_count >= settings.MAX_RECOVERY_ATTEMPTS:
            return (
                InterventionType.NO_ACTION,
                f"MAX_RETRY_EXCEEDED: Transaction already attempted {txn.retry_attempt_count} times (limit {settings.MAX_RECOVERY_ATTEMPTS}).",
                0,
            )

        # Invariant 2: High Value Circuit Breaker (>= ₹50,000 requires human merchant escalation)
        if txn.amount_inr >= settings.HIGH_VALUE_THRESHOLD_INR:
            return (
                InterventionType.MANUAL_ESCALATION,
                f"HIGH_VALUE_GUARD: Amount INR {txn.amount_inr:,.2f} exceeds automated threshold INR {settings.HIGH_VALUE_THRESHOLD_INR:,.2f}. Escalating to merchant operations.",
                0,
            )

        return None, None, 0

    @classmethod
    def get_forbidden_actions(cls, txn: PaymentTransaction) -> Set[InterventionType]:
        """
        Returns the set of actions that are physically impossible or prohibited for this incident.
        """
        forbidden = set()

        # 1. 3DS Authentication Failure -> Auto-retry without customer OTP is impossible
        if txn.failure_category == FailureCategory.AUTH_FAILED_3DS:
            forbidden.add(InterventionType.SMART_RETRY_IMMEDIATE)
            forbidden.add(InterventionType.SMART_RETRY_DELAYED)

        # 2. Mandate Expired -> Auto-retry on invalid/expired token is rejected by NPCI/Issuer
        elif txn.failure_category == FailureCategory.MANDATE_EXPIRED:
            forbidden.add(InterventionType.SMART_RETRY_IMMEDIATE)
            forbidden.add(InterventionType.SMART_RETRY_DELAYED)

        # 3. Bank Downtime during midnight maintenance -> Immediate retry is guaranteed to fail
        elif txn.failure_category == FailureCategory.BANK_DOWNTIME:
            if txn.hour_of_day in [23, 0, 1, 2, 3]:
                forbidden.add(InterventionType.SMART_RETRY_IMMEDIATE)

        # 4. Micro/Small transactions (< INR 5,000) -> Manual escalation is cost-inefficient
        if txn.amount_inr < 5000.0:
            forbidden.add(InterventionType.MANUAL_ESCALATION)

        return forbidden
