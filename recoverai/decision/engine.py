"""
Expected Value Decision Engine for RecoverAI.

Implements Tiered Action Selection driven by Calibrated ML Probabilities and Expected Value:
- Tier 1 (Automated Retries & Backoff): Zero-friction, low-cost (₹0.50) automated retries.
- Tier 2 (Frictionless Rail Switch): Low-friction (₹1.50), low-cost (₹0.40) UPI Intent switches.
- Tier 3 (Conversational 1-Click Links): Interactive WhatsApp/SMS payment links factoring in customer LTV friction.
- Tier 4 (White-Glove Concierge Escalation): Reserved for high-expected-yield (EV >= ₹12,000) or VIP accounts (₹50 ops cost).
"""

import logging
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from recoverai.data.schemas import (
    PaymentTransaction,
    InterventionType,
    RecoveryDecision,
    FailureCategory,
    CustomerTier,
    CommunicationChannel,
)
from recoverai.decision.rules import BusinessRuleEngine
from recoverai.features.pipeline import FeaturePipeline
from recoverai.models.recovery_classifier import CalibratedRecoveryClassifier
from recoverai.config.settings import settings

logger = logging.getLogger("recoverai.decision.engine")

# Base Action Specs & Operational Costs
INTERVENTION_SPECS = {
    InterventionType.SMART_RETRY_DELAYED: {
        "tier": 1,
        "cost_inr": settings.COST_AUTO_RETRY_INR,
        "friction_inr": 0.0,
        "supported_categories": [
            FailureCategory.BANK_DOWNTIME,
            FailureCategory.NETWORK_ERROR,
        ],
        "prob_multiplier": 1.15,
    },
    InterventionType.SMART_RETRY_IMMEDIATE: {
        "tier": 1,
        "cost_inr": settings.COST_AUTO_RETRY_INR,
        "friction_inr": 0.0,
        "supported_categories": [FailureCategory.NETWORK_ERROR],
        "prob_multiplier": 1.00,
    },
    InterventionType.UPI_INTENT_SWITCH: {
        "tier": 2,
        "cost_inr": settings.COST_UPI_INTENT_INR,
        "friction_inr": 1.50,
        "supported_categories": [
            FailureCategory.AUTH_FAILED_3DS,
            FailureCategory.CARD_LIMIT_EXCEEDED,
            FailureCategory.INSUFFICIENT_FUNDS,
        ],
        "prob_multiplier": 1.08,
    },
    InterventionType.WHATSAPP_PAY_LINK: {
        "tier": 3,
        "cost_inr": settings.COST_WHATSAPP_INR,
        "friction_inr": settings.CUSTOMER_FRICTION_PENALTY_INR,
        "supported_categories": [
            FailureCategory.AUTH_FAILED_3DS,
            FailureCategory.MANDATE_EXPIRED,
            FailureCategory.INSUFFICIENT_FUNDS,
        ],
        "prob_multiplier": 1.12,
    },
    InterventionType.SMS_PAY_LINK: {
        "tier": 3,
        "cost_inr": settings.COST_SMS_INR,
        "friction_inr": settings.CUSTOMER_FRICTION_PENALTY_INR,
        "supported_categories": [
            FailureCategory.AUTH_FAILED_3DS,
            FailureCategory.MANDATE_EXPIRED,
            FailureCategory.INSUFFICIENT_FUNDS,
        ],
        "prob_multiplier": 0.85,
    },
    InterventionType.MANUAL_ESCALATION: {
        "tier": 4,
        "cost_inr": settings.COST_MANUAL_ESCALATION_INR,
        "friction_inr": 0.0,
        "supported_categories": [
            FailureCategory.CARD_LIMIT_EXCEEDED,
            FailureCategory.BANK_DOWNTIME,
            FailureCategory.AUTH_FAILED_3DS,
        ],
        "prob_multiplier": 1.18,
    },
    InterventionType.NO_ACTION: {
        "tier": 0,
        "cost_inr": 0.0,
        "friction_inr": 0.0,
        "supported_categories": list(FailureCategory),
        "prob_multiplier": 0.0,
    },
}


class ExpectedValueDecisionEngine:
    """Selects the mathematically optimal intervention tier maximizing Net Expected ROI."""

    def __init__(
        self,
        classifier: Optional[CalibratedRecoveryClassifier] = None,
        feature_pipeline: Optional[FeaturePipeline] = None,
    ):
        self.classifier = classifier or CalibratedRecoveryClassifier()
        self.feature_pipeline = feature_pipeline or FeaturePipeline()

    def decide_intervention(self, txn: PaymentTransaction) -> RecoveryDecision:
        """
        Runs full tiered decision pipeline:
        1. Checks Hard Safety Invariants (Circuit Breakers / Rate Limits)
        2. Computes Base Calibrated Recovery Probability from ML Model
        3. Applies Category-specific Physics, Channel Affinity & LTV Friction Scaling
        4. Selects Action Tier maximizing Net Expected Monetary Value
        """
        amount = txn.amount_inr

        # Step 1: Check Deterministic Hard Invariants
        forced_action, rule_reason, forced_delay = BusinessRuleEngine.apply_hard_rules(txn)
        if forced_action is not None:
            spec = INTERVENTION_SPECS[forced_action]
            action_cost = spec["cost_inr"]
            friction = spec["friction_inr"]
            prob = 0.88 if forced_action == InterventionType.MANUAL_ESCALATION else 0.0
            expected_gross = prob * amount
            net_roi = expected_gross - action_cost - friction

            return RecoveryDecision(
                transaction_id=txn.transaction_id,
                recommended_action=forced_action,
                recommended_delay_hours=forced_delay,
                estimated_recovery_probability=round(prob, 4),
                expected_recovery_value_inr=round(expected_gross, 2),
                intervention_cost_inr=round(action_cost + friction, 2),
                net_expected_roi_inr=round(net_roi, 2),
                action_confidence=round(prob, 4),
                safety_rule_applied=rule_reason,
                reasoning_summary=f"Enforced by safety invariant: {rule_reason}",
            )

        # Step 2: Extract Features & Base Calibrated Probability from ML Model
        txn_dict = txn.model_dump()
        try:
            X_single = self.feature_pipeline.transform_single(txn_dict)
            base_prob = float(self.classifier.predict_recovery_probability(X_single)[0])
        except Exception as e:
            logger.warning(f"Fallback to heuristic probability due to feature transform error: {e}")
            base_prob = 0.65

        forbidden_actions = BusinessRuleEngine.get_forbidden_actions(txn)

        # Dynamic customer friction scaling: Higher-LTV relationships have higher customer fatigue cost
        ltv_friction_scale = 1.0 + min(txn.customer_ltv_inr / 40000.0, 2.0)

        best_action = InterventionType.NO_ACTION
        best_net_roi = -float("inf")
        best_prob = 0.0
        best_cost = 0.0
        best_delay = 0

        # Evaluate all candidate actions
        for action, spec in INTERVENTION_SPECS.items():
            if action == InterventionType.NO_ACTION or action in forbidden_actions:
                continue
            if txn.failure_category not in spec["supported_categories"]:
                continue

            action_prob = min(max(base_prob * spec["prob_multiplier"], 0.05), 0.96)
            action_cost = spec["cost_inr"]
            action_friction = spec["friction_inr"] * (ltv_friction_scale if "PAY_LINK" in action.value else 1.0)
            delay = 0

            # --- Channel Physics & Tier Assignment Logic ---
            if txn.failure_category == FailureCategory.BANK_DOWNTIME:
                if action == InterventionType.SMART_RETRY_DELAYED:
                    action_prob = min(base_prob * 1.15, 0.95)
                    delay = max(1, 5 - txn.hour_of_day) if txn.hour_of_day in [23, 0, 1, 2, 3] else 1
                elif action == InterventionType.SMART_RETRY_IMMEDIATE:
                    action_prob = 0.07

            elif txn.failure_category == FailureCategory.AUTH_FAILED_3DS:
                if action == InterventionType.WHATSAPP_PAY_LINK:
                    boost = 0.08 if txn.preferred_channel == CommunicationChannel.WHATSAPP else 0.0
                    action_prob = min(base_prob * 1.08 + boost, 0.92)
                elif action == InterventionType.SMS_PAY_LINK:
                    boost = 0.04 if txn.preferred_channel == CommunicationChannel.SMS else -0.10
                    action_prob = max(base_prob * 0.85 + boost, 0.40)
                elif action == InterventionType.UPI_INTENT_SWITCH:
                    action_prob = min(base_prob * 1.05, 0.88)

            elif txn.failure_category == FailureCategory.MANDATE_EXPIRED:
                if action == InterventionType.WHATSAPP_PAY_LINK:
                    boost = 0.06 if txn.preferred_channel == CommunicationChannel.WHATSAPP else 0.0
                    action_prob = min(base_prob * 1.06 + boost, 0.90)
                elif action == InterventionType.SMS_PAY_LINK:
                    action_prob = max(base_prob * 0.80, 0.40)

            elif txn.failure_category == FailureCategory.INSUFFICIENT_FUNDS:
                delay = 24 if txn.day_of_month >= 28 else 48
                if action == InterventionType.WHATSAPP_PAY_LINK:
                    action_prob = min(base_prob * 1.05, 0.85)
                elif action == InterventionType.SMS_PAY_LINK:
                    action_prob = max(base_prob * 0.80, 0.35)

            elif txn.failure_category == FailureCategory.CARD_LIMIT_EXCEEDED:
                # Tier 4 Concierge Manual Escalation: only deploy when expected gross yield >= ₹12,000 or VIP account
                if action == InterventionType.MANUAL_ESCALATION:
                    if base_prob * amount >= 12000.0 or txn.customer_tier == CustomerTier.VIP:
                        action_prob = 0.88
                    else:
                        # Negative incremental ROI vs ₹50 human ops cost on small tickets
                        action_prob = 0.50
                elif action == InterventionType.UPI_INTENT_SWITCH:
                    action_prob = min(base_prob * 1.02, 0.80)

            elif txn.failure_category == FailureCategory.NETWORK_ERROR:
                if action == InterventionType.SMART_RETRY_IMMEDIATE:
                    action_prob = min(base_prob * 1.10, 0.92)

            gross_expected = action_prob * amount
            net_expected = gross_expected - action_cost - action_friction

            if net_expected > best_net_roi:
                best_net_roi = net_expected
                best_action = action
                best_prob = action_prob
                best_cost = action_cost + action_friction
                best_delay = delay

        # If net expected value is below zero, suppress
        if best_net_roi < settings.MIN_EXPECTED_VALUE_INR or best_action == InterventionType.NO_ACTION:
            return RecoveryDecision(
                transaction_id=txn.transaction_id,
                recommended_action=InterventionType.NO_ACTION,
                recommended_delay_hours=0,
                estimated_recovery_probability=0.0,
                expected_recovery_value_inr=0.0,
                intervention_cost_inr=0.0,
                net_expected_roi_inr=0.0,
                action_confidence=1.0,
                safety_rule_applied="MIN_ROI_NOT_MET",
                reasoning_summary="Expected recovery value is below operational cost. Suppressed to prevent negative ROI.",
            )

        return RecoveryDecision(
            transaction_id=txn.transaction_id,
            recommended_action=best_action,
            recommended_delay_hours=best_delay,
            estimated_recovery_probability=round(best_prob, 4),
            expected_recovery_value_inr=round(best_prob * amount, 2),
            intervention_cost_inr=round(best_cost, 2),
            net_expected_roi_inr=round(best_net_roi, 2),
            action_confidence=round(best_prob, 4),
            safety_rule_applied=None,
            reasoning_summary=(
                f"Selected Tiered Action {best_action.value} (Net ROI: INR {best_net_roi:,.2f}, "
                f"P(Recov): {best_prob * 100:.1f}%, Delay: {best_delay}h)."
            ),
        )
