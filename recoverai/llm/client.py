"""
LLM Reasoning & Explanation Client with Structured Pydantic Output & Deterministic Fallback.
"""

import json
import logging
from typing import Optional, Dict, Any
from pydantic import ValidationError

from recoverai.data.schemas import (
    PaymentTransaction,
    RecoveryDecision,
    LLMExplanation,
    CommunicationChannel,
    FailureCategory,
    InterventionType,
)
from recoverai.llm.prompts import (
    SYSTEM_DIAGNOSIS_PROMPT,
    RECOVERY_EXPLANATION_USER_PROMPT,
)
from recoverai.config.settings import settings

logger = logging.getLogger("recoverai.llm.client")


class DeterministicFallbackReasoningEngine:
    """Provides instant, offline, deterministic reasoning without requiring external API keys."""

    @staticmethod
    def generate_explanation(txn: PaymentTransaction, decision: RecoveryDecision) -> LLMExplanation:
        """Constructs calibrated diagnosis, merchant guidance, and customer copy deterministically."""
        amount_fmt = f"INR {txn.amount_inr:,.2f}"
        link = f"https://rzp.io/i/{txn.transaction_id}"
        
        # 1. Root Cause Diagnosis
        if txn.failure_category == FailureCategory.BANK_DOWNTIME:
            diagnosis = (
                f"Transient server timeout at {txn.bank_code} during peak routing. "
                f"Customer account was not debited. The error ({txn.gateway_error_code}) is on the issuer switch side."
            )
            recommendation = (
                f"Schedule an automated smart retry after the {decision.recommended_delay_hours}h backoff window. "
                f"Do not prompt the customer manually to prevent double-debit panic."
            )
            message = (
                f"Hi there! Your payment of {amount_fmt} to {txn.merchant_category} was paused due to a temporary "
                f"{txn.bank_code} bank server delay. We will safely retry this automatically. No action needed right now!"
            )
            urgency = "LOW"

        elif txn.failure_category == FailureCategory.INSUFFICIENT_FUNDS:
            diagnosis = (
                f"Decline code {txn.gateway_error_code}: Insufficient balance in customer account / card credit line limit."
            )
            recommendation = (
                f"Send a gentle 1-click payment link on {txn.preferred_channel.value}. Avoid aggressive dunning for VIP accounts."
            )
            message = (
                f"Hi! We were unable to complete your payment of {amount_fmt}. "
                f"You can quickly complete it or choose a different payment method here: {link}"
            )
            urgency = "MEDIUM"

        elif txn.failure_category == FailureCategory.AUTH_FAILED_3DS:
            diagnosis = (
                f"3DS Authentication dropped ({txn.gateway_error_code}). Customer OTP expired or authentication window was closed."
            )
            recommendation = (
                f"Dispatch instant 1-click WhatsApp/UPI pay link. Automated retry is impossible without user OTP."
            )
            message = (
                f"Hello! It looks like your payment of {amount_fmt} was interrupted during OTP verification. "
                f"Click here to instantly complete your purchase with UPI / Card: {link}"
            )
            urgency = "HIGH"

        elif txn.failure_category == FailureCategory.MANDATE_EXPIRED:
            diagnosis = (
                f"Recurring e-mandate invalid or expired ({txn.gateway_error_code}). Subscription auto-debit rejected by NPCI/bank."
            )
            recommendation = (
                f"Send recurring mandate re-authorization link to prevent involuntary subscription cancellation."
            )
            message = (
                f"Important: Your recurring subscription payment of {amount_fmt} could not be processed because "
                f"your payment mandate needs a quick renewal. Re-authorize securely in 1-click: {link}"
            )
            urgency = "HIGH"

        elif txn.failure_category == FailureCategory.CARD_LIMIT_EXCEEDED:
            diagnosis = (
                f"Card limit exceeded ({txn.gateway_error_code}). Transaction amount {amount_fmt} exceeds online transaction ceiling."
            )
            recommendation = (
                f"Offer instant UPI Intent / NetBanking alternative rails to bypass card limit restrictions."
            )
            message = (
                f"Hi! Your card payment of {amount_fmt} reached its daily bank limit. "
                f"You can easily switch to UPI or NetBanking to complete your order: {link}"
            )
            urgency = "MEDIUM"

        else:
            diagnosis = f"Transient gateway connectivity glitch ({txn.gateway_error_code})."
            recommendation = "Execute immediate smart retry."
            message = f"Payment status update: Your transaction of {amount_fmt} is being reprocessed."
            urgency = "LOW"

        channel = (
            CommunicationChannel.NONE
            if decision.recommended_action in [InterventionType.SMART_RETRY_DELAYED, InterventionType.SMART_RETRY_IMMEDIATE, InterventionType.NO_ACTION]
            else txn.preferred_channel
        )

        return LLMExplanation(
            root_cause_diagnosis=diagnosis,
            merchant_recommendation=recommendation,
            customer_recovery_message=message,
            channel_selected=channel,
            urgency_level=urgency,
        )


class LLMReasoningClient:
    """Orchestrates LLM synthesis with strict schema validation and graceful fallback."""

    def __init__(self):
        self.provider = settings.LLM_PROVIDER
        self.openai_key = settings.OPENAI_API_KEY
        self.gemini_key = settings.GEMINI_API_KEY

    def explain_and_draft_recovery(
        self,
        txn: PaymentTransaction,
        decision: RecoveryDecision,
    ) -> LLMExplanation:
        """
        Generates structured explanation. Tries live LLM if configured; otherwise uses deterministic fallback.
        """
        if self.provider == "openai" and self.openai_key:
            try:
                return self._call_openai(txn, decision)
            except Exception as e:
                logger.warning(f"OpenAI call failed ({e}). Falling back to deterministic reasoning engine.")
        
        elif self.provider == "gemini" and self.gemini_key:
            try:
                return self._call_gemini(txn, decision)
            except Exception as e:
                logger.warning(f"Gemini call failed ({e}). Falling back to deterministic reasoning engine.")

        # Default fallback
        return DeterministicFallbackReasoningEngine.generate_explanation(txn, decision)

    def _call_openai(self, txn: PaymentTransaction, decision: RecoveryDecision) -> LLMExplanation:
        """Calls OpenAI with JSON mode and Pydantic validation."""
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_key)

        user_content = RECOVERY_EXPLANATION_USER_PROMPT.format(
            merchant_name=txn.merchant_id,
            merchant_category=txn.merchant_category,
            transaction_id=txn.transaction_id,
            amount_inr=txn.amount_inr,
            customer_id=txn.customer_id,
            customer_tier=txn.customer_tier.value,
            customer_ltv=txn.customer_ltv_inr,
            preferred_channel=txn.preferred_channel.value,
            payment_method=txn.payment_method.value,
            bank_code=txn.bank_code,
            card_network=txn.card_network,
            failure_category=txn.failure_category.value,
            gateway_error_code=txn.gateway_error_code,
            gateway_error_description=txn.gateway_error_description,
            retry_count=txn.retry_attempt_count,
            recommended_action=decision.recommended_action.value,
            recommended_delay_hours=decision.recommended_delay_hours,
            recovery_prob_pct=decision.estimated_recovery_probability * 100.0,
            net_roi=decision.net_expected_roi_inr,
        )

        response = client.chat.completions.create(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_DIAGNOSIS_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        return LLMExplanation(**data)

    def _call_gemini(self, txn: PaymentTransaction, decision: RecoveryDecision) -> LLMExplanation:
        """Calls Google Gemini with structured generation."""
        import google.generativeai as genai
        genai.configure(api_key=self.gemini_key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json"},
        )
        user_content = RECOVERY_EXPLANATION_USER_PROMPT.format(
            merchant_name=txn.merchant_id,
            merchant_category=txn.merchant_category,
            transaction_id=txn.transaction_id,
            amount_inr=txn.amount_inr,
            customer_id=txn.customer_id,
            customer_tier=txn.customer_tier.value,
            customer_ltv=txn.customer_ltv_inr,
            preferred_channel=txn.preferred_channel.value,
            payment_method=txn.payment_method.value,
            bank_code=txn.bank_code,
            card_network=txn.card_network,
            failure_category=txn.failure_category.value,
            gateway_error_code=txn.gateway_error_code,
            gateway_error_description=txn.gateway_error_description,
            retry_count=txn.retry_attempt_count,
            recommended_action=decision.recommended_action.value,
            recommended_delay_hours=decision.recommended_delay_hours,
            recovery_prob_pct=decision.estimated_recovery_probability * 100.0,
            net_roi=decision.net_expected_roi_inr,
        )
        resp = model.generate_content(f"{SYSTEM_DIAGNOSIS_PROMPT}\n\n{user_content}")
        data = json.loads(resp.text)
        return LLMExplanation(**data)
