"""Prompt definitions and system directives for RecoverAI LLM layer."""

SYSTEM_DIAGNOSIS_PROMPT = """You are the Senior AI Revenue Recovery Specialist for Razorpay Merchants.
Your mission is to analyze failed payment incidents, provide clear root-cause diagnoses for merchant finance operations,
and draft empathetic, high-converting customer recovery communications.

STRICT OPERATIONAL GUIDELINES:
1. NEVER invent or guess arbitrary bank codes; only analyze provided data.
2. Formulate empathetic, clear, and non-accusatory customer recovery messages. Never imply the customer is at fault.
3. Include clear 1-click payment resolution instructions.
4. Output MUST conform strictly to the required JSON schema.
"""

RECOVERY_EXPLANATION_USER_PROMPT = """Analyze the following payment failure incident:

Merchant: {merchant_name} ({merchant_category})
Transaction ID: {transaction_id}
Amount: INR {amount_inr:,.2f}
Customer ID: {customer_id} (Tier: {customer_tier}, LTV: INR {customer_ltv:,.2f})
Preferred Channel: {preferred_channel}
Payment Method: {payment_method}
Bank / Issuer: {bank_code} ({card_network})
Failure Category: {failure_category}
Gateway Error Code: {gateway_error_code} - {gateway_error_description}
Retry Count: {retry_count}

Selected Intervention: {recommended_action}
Recommended Delay: {recommended_delay_hours} hours
Estimated Recovery Probability: {recovery_prob_pct:.1f}%
Expected Net ROI: INR {net_roi:,.2f}

Provide your response in JSON with the following keys:
- root_cause_diagnosis: (string) Concise technical explanation of why the payment failed.
- merchant_recommendation: (string) Operational advice for the merchant's finance/support team.
- customer_recovery_message: (string) Empathetic, personalized notification tailored for {preferred_channel} with a payment link placeholder [https://rzp.io/i/{transaction_id}].
- channel_selected: (string) WHATSAPP, SMS, EMAIL, or NONE.
- urgency_level: (string) LOW, MEDIUM, HIGH, or CRITICAL.
"""
