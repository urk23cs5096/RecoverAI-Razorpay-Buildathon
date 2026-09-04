import math
from datetime import datetime
from enum import Enum
from typing import Optional, List, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator


class PaymentMethod(str, Enum):
    UPI = "UPI"
    CARD = "CARD"
    NETBANKING = "NETBANKING"
    EMANDATE_CARD = "EMANDATE_CARD"
    EMANDATE_UPI = "EMANDATE_UPI"


class FailureCategory(str, Enum):
    BANK_DOWNTIME = "BANK_DOWNTIME"
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    AUTH_FAILED_3DS = "AUTH_FAILED_3DS"
    MANDATE_EXPIRED = "MANDATE_EXPIRED"
    CARD_LIMIT_EXCEEDED = "CARD_LIMIT_EXCEEDED"
    NETWORK_ERROR = "NETWORK_ERROR"


class CustomerTier(str, Enum):
    VIP = "VIP"
    REGULAR = "REGULAR"
    NEW = "NEW"
    AT_RISK = "AT_RISK"


class CommunicationChannel(str, Enum):
    WHATSAPP = "WHATSAPP"
    SMS = "SMS"
    EMAIL = "EMAIL"
    NONE = "NONE"


class InterventionType(str, Enum):
    SMART_RETRY_DELAYED = "SMART_RETRY_DELAYED"
    SMART_RETRY_IMMEDIATE = "SMART_RETRY_IMMEDIATE"
    WHATSAPP_PAY_LINK = "WHATSAPP_PAY_LINK"
    SMS_PAY_LINK = "SMS_PAY_LINK"
    UPI_INTENT_SWITCH = "UPI_INTENT_SWITCH"
    MANUAL_ESCALATION = "MANUAL_ESCALATION"
    NO_ACTION = "NO_ACTION"


class AgentState(str, Enum):
    DETECTED = "DETECTED"
    DIAGNOSED = "DIAGNOSED"
    DECIDED = "DECIDED"
    ACTION_SCHEDULED = "ACTION_SCHEDULED"
    ACTION_EXECUTED = "ACTION_EXECUTED"
    VERIFIED = "VERIFIED"
    RECOVERED = "RECOVERED"
    FAILED_TERMINAL = "FAILED_TERMINAL"
    ESCALATED = "ESCALATED"


class PaymentTransaction(BaseModel):
    """Raw or Ingested Payment Transaction Event."""
    model_config = ConfigDict(from_attributes=True)

    @model_validator(mode="before")
    @classmethod
    def sanitize_nan_values(cls, data: Any) -> Any:
        if isinstance(data, dict):
            cleaned = {}
            for k, v in data.items():
                if isinstance(v, float) and math.isnan(v):
                    cleaned[k] = None
                else:
                    cleaned[k] = v
            return cleaned
        return data

    transaction_id: str = Field(..., description="Unique transaction ID (e.g. txn_1001)")
    merchant_id: str = Field(..., description="Merchant identifier (e.g. merch_saas_01)")
    merchant_category: str = Field(..., description="Merchant vertical: SaaS, D2C, EdTech, B2B")
    customer_id: str = Field(..., description="Customer identifier")
    customer_tier: CustomerTier = Field(default=CustomerTier.REGULAR)
    customer_ltv_inr: float = Field(..., ge=0.0, description="Historical customer LTV in INR")
    preferred_channel: CommunicationChannel = Field(default=CommunicationChannel.WHATSAPP)
    
    amount_inr: float = Field(..., gt=0.0, description="Transaction amount in INR")
    currency: str = Field(default="INR")
    payment_method: PaymentMethod
    bank_code: str = Field(..., description="Bank identifier: HDFC, ICICI, SBI, AXIS, etc.")
    card_network: Optional[str] = Field(default="NONE", description="VISA, MASTERCARD, RUPAY, AMEX, NONE")
    is_recurring: bool = Field(default=False, description="Whether this is a recurring subscription")
    mandate_id: Optional[str] = Field(default=None)
    
    timestamp: datetime = Field(..., description="Timestamp of payment event")
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    day_of_month: int = Field(default=15, ge=1, le=31)
    
    # Failure information
    is_failed: bool = Field(default=True)
    gateway_error_code: str = Field(..., description="Gateway error code: U19, 51, 91, 3DS_TIMEOUT, etc.")
    gateway_error_description: str
    failure_category: FailureCategory
    retry_attempt_count: int = Field(default=0, ge=0)
    last_retry_timestamp: Optional[datetime] = None


class EnrichedFeatureRecord(PaymentTransaction):
    """Enriched transaction record with engineered features for ML inference."""
    bank_downtime_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    customer_failure_history_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    amount_to_ltv_ratio: float = Field(default=0.0, ge=0.0)
    is_high_risk_hour: int = Field(default=0, ge=0, le=1)
    is_salary_cycle_day: int = Field(default=0, ge=0, le=1)
    payment_method_reliability_score: float = Field(default=0.85, ge=0.0, le=1.0)


class RecoveryDecision(BaseModel):
    """Output of the Expected Value Recovery Decision Engine."""
    transaction_id: str
    recommended_action: InterventionType
    recommended_delay_hours: int = 0
    estimated_recovery_probability: float = Field(..., ge=0.0, le=1.0)
    expected_recovery_value_inr: float
    intervention_cost_inr: float
    net_expected_roi_inr: float
    action_confidence: float = Field(..., ge=0.0, le=1.0)
    safety_rule_applied: Optional[str] = None
    reasoning_summary: str


class LLMExplanation(BaseModel):
    """Structured reasoning output from the LLM Layer."""
    root_cause_diagnosis: str = Field(..., description="Plain-language explanation of why payment failed")
    merchant_recommendation: str = Field(..., description="Actionable summary for merchant ops team")
    customer_recovery_message: str = Field(..., description="Empathetic, clear recovery notification text")
    channel_selected: CommunicationChannel
    urgency_level: str = Field(default="MEDIUM", description="LOW, MEDIUM, HIGH, CRITICAL")


class AuditLogRecord(BaseModel):
    """Immutable audit record persisted for each agent lifecycle transition."""
    audit_id: Optional[int] = None
    transaction_id: str
    timestamp: datetime
    previous_state: AgentState
    current_state: AgentState
    action_type: InterventionType
    actor: str = Field(default="RecoverAI_Agent")
    model_version: str = "v1.0.0"
    model_prediction: float
    cost_incurred_inr: float = 0.0
    amount_recovered_inr: float = 0.0
    details_json: dict = Field(default_factory=dict)
