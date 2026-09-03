"""
Realistic Stochastic Recovery Simulator for Indian Digital Payment Rails.

Models bank switch restoration curves, customer payment link conversions,
UPI Intent completion rates, and human escalation outcomes based on empirical priors.
"""

import random
import logging
from typing import Dict, Any, Tuple
from recoverai.data.schemas import (
    PaymentTransaction,
    InterventionType,
    FailureCategory,
    CustomerTier,
    CommunicationChannel,
)
from recoverai.config.settings import settings

logger = logging.getLogger("recoverai.simulator.engine")


class RecoverySimulatorEngine:
    """Simulates realistic gateway and customer behavioral responses to recovery actions."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        random.seed(seed)

    def simulate_outcome(
        self,
        txn: PaymentTransaction,
        action: InterventionType,
        delay_hours: int = 0,
    ) -> Dict[str, Any]:
        """
        Simulates the execution outcome for a specific intervention.
        
        Returns:
            Dict containing:
            - is_recovered: bool
            - amount_recovered_inr: float
            - operational_cost_inr: float
            - friction_penalty_inr: float
            - response_channel: str
            - time_to_recovery_hours: float
            - outcome_description: str
        """
        if action == InterventionType.NO_ACTION:
            return {
                "is_recovered": False,
                "amount_recovered_inr": 0.0,
                "operational_cost_inr": 0.0,
                "friction_penalty_inr": 0.0,
                "response_channel": "NONE",
                "time_to_recovery_hours": 0.0,
                "outcome_description": "No intervention dispatched. Revenue lost.",
            }

        # Operational costs
        cost_map = {
            InterventionType.SMART_RETRY_DELAYED: settings.COST_AUTO_RETRY_INR,
            InterventionType.SMART_RETRY_IMMEDIATE: settings.COST_AUTO_RETRY_INR,
            InterventionType.WHATSAPP_PAY_LINK: settings.COST_WHATSAPP_INR,
            InterventionType.SMS_PAY_LINK: settings.COST_SMS_INR,
            InterventionType.UPI_INTENT_SWITCH: settings.COST_UPI_INTENT_INR,
            InterventionType.MANUAL_ESCALATION: settings.COST_MANUAL_ESCALATION_INR,
        }
        operational_cost = cost_map.get(action, 0.50)
        friction_penalty = settings.CUSTOMER_FRICTION_PENALTY_INR if "PAY_LINK" in action.value else 0.0

        # Base success probability based on action and failure physics
        success_prob = self._compute_simulation_probability(txn, action, delay_hours)
        is_success = self.rng.random() < success_prob

        time_to_recovery = float(delay_hours) + self.rng.uniform(0.1, 1.5) if is_success else 0.0
        amount_recovered = txn.amount_inr if is_success else 0.0

        if is_success:
            desc = f"Payment successfully recovered via {action.value} after {time_to_recovery:.1f}h."
        else:
            desc = f"Recovery attempt via {action.value} failed. Customer/Bank did not settle."

        return {
            "is_recovered": is_success,
            "amount_recovered_inr": round(amount_recovered, 2),
            "operational_cost_inr": round(operational_cost, 2),
            "friction_penalty_inr": round(friction_penalty, 2),
            "response_channel": action.value,
            "time_to_recovery_hours": round(time_to_recovery, 2),
            "outcome_description": desc,
        }

    def _compute_simulation_probability(
        self,
        txn: PaymentTransaction,
        action: InterventionType,
        delay_hours: int,
    ) -> float:
        """Calculates realistic empirical recovery likelihood."""
        cat = txn.failure_category
        tier_boost = 0.08 if txn.customer_tier == CustomerTier.VIP else 0.0

        # Scenario 1: Bank Downtime
        if cat == FailureCategory.BANK_DOWNTIME:
            if action == InterventionType.SMART_RETRY_DELAYED and delay_hours >= 1:
                return 0.88 + tier_boost
            elif action == InterventionType.SMART_RETRY_IMMEDIATE:
                return 0.07  # Immediate retry while bank switch is still down fails 93% of the time
            elif "PAY_LINK" in action.value:
                return 0.45  # Customer might pay with a different bank card

        # Scenario 2: Insufficient Funds
        elif cat == FailureCategory.INSUFFICIENT_FUNDS:
            if action == InterventionType.WHATSAPP_PAY_LINK and delay_hours >= 24:
                return 0.68 + tier_boost
            elif action == InterventionType.SMART_RETRY_IMMEDIATE:
                return 0.04  # Account still empty
            elif action == InterventionType.UPI_INTENT_SWITCH:
                return 0.60

        # Scenario 3: 3DS Authentication Failure
        elif cat == FailureCategory.AUTH_FAILED_3DS:
            if action in [InterventionType.WHATSAPP_PAY_LINK, InterventionType.UPI_INTENT_SWITCH]:
                return 0.80 + tier_boost
            elif action in [InterventionType.SMART_RETRY_DELAYED, InterventionType.SMART_RETRY_IMMEDIATE]:
                return 0.00  # Impossible to auto-retry without OTP

        # Scenario 4: Mandate Expired
        elif cat == FailureCategory.MANDATE_EXPIRED:
            if action == InterventionType.WHATSAPP_PAY_LINK:
                return 0.74 + tier_boost
            elif "RETRY" in action.value:
                return 0.00  # Gateway hard decline

        # Scenario 5: Card Limit Exceeded
        elif cat == FailureCategory.CARD_LIMIT_EXCEEDED:
            if action == InterventionType.UPI_INTENT_SWITCH:
                return 0.72 + tier_boost
            elif action == InterventionType.MANUAL_ESCALATION:
                return 0.85
            elif "RETRY" in action.value:
                return 0.02

        # Scenario 6: Network Error
        elif cat == FailureCategory.NETWORK_ERROR:
            if action in [InterventionType.SMART_RETRY_IMMEDIATE, InterventionType.SMART_RETRY_DELAYED]:
                return 0.85
            return 0.70

        # High value manual escalation general boost
        if action == InterventionType.MANUAL_ESCALATION:
            return 0.84 + tier_boost

        return 0.50
