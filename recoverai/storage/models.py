"""SQLAlchemy Database Models for RecoverAI."""

from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    ForeignKey,
    JSON,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


def utcnow() -> datetime:
    """Returns current UTC timestamp as a timezone-naive datetime for clean DB storage."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class TransactionDB(Base):
    """Database representation of a payment transaction."""
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), unique=True, index=True, nullable=False)
    order_id = Column(String(64), nullable=True)
    merchant_id = Column(String(64), index=True, nullable=False)
    merchant_category = Column(String(64), nullable=False)
    customer_id = Column(String(64), index=True, nullable=False)
    customer_tier = Column(String(32), default="REGULAR")
    customer_ltv_inr = Column(Float, default=0.0)
    preferred_channel = Column(String(32), default="WHATSAPP")
    amount_inr = Column(Float, nullable=False)
    currency = Column(String(10), default="INR")
    payment_method = Column(String(32), nullable=False)
    bank_code = Column(String(32), nullable=False)
    card_network = Column(String(32), default="NONE")
    is_recurring = Column(Boolean, default=False)
    mandate_id = Column(String(64), nullable=True)
    failure_category = Column(String(64), nullable=False)
    gateway_error_code = Column(String(32), nullable=False)
    gateway_error_description = Column(String(255), nullable=False)
    retry_attempt_count = Column(Integer, default=0)
    current_agent_state = Column(String(32), default="DETECTED")
    is_recovered = Column(Boolean, default=False)
    amount_recovered_inr = Column(Float, default=0.0)
    created_at = Column(DateTime, default=utcnow)
    updated_at = Column(DateTime, default=utcnow, onupdate=utcnow)

    # Relationships
    decisions = relationship("DecisionDB", back_populates="transaction", cascade="all, delete-orphan")
    audit_events = relationship("AuditEventDB", back_populates="transaction", cascade="all, delete-orphan")


class DecisionDB(Base):
    """Database record of an ML/Decision Engine evaluation."""
    __tablename__ = "recovery_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), ForeignKey("transactions.transaction_id"), index=True, nullable=False)
    recommended_action = Column(String(64), nullable=False)
    recommended_delay_hours = Column(Integer, default=0)
    estimated_recovery_probability = Column(Float, nullable=False)
    expected_recovery_value_inr = Column(Float, nullable=False)
    intervention_cost_inr = Column(Float, nullable=False)
    net_expected_roi_inr = Column(Float, nullable=False)
    action_confidence = Column(Float, default=1.0)
    safety_rule_applied = Column(String(255), nullable=True)
    reasoning_summary = Column(Text, nullable=False)
    root_cause_diagnosis = Column(Text, nullable=True)
    merchant_recommendation = Column(Text, nullable=True)
    customer_recovery_message = Column(Text, nullable=True)
    created_at = Column(DateTime, default=utcnow)

    transaction = relationship("TransactionDB", back_populates="decisions")


class AuditEventDB(Base):
    """Immutable audit event log for compliance, debugging, and explainability."""
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    transaction_id = Column(String(64), ForeignKey("transactions.transaction_id"), index=True, nullable=False)
    timestamp = Column(DateTime, default=utcnow, nullable=False)
    previous_state = Column(String(32), nullable=False)
    current_state = Column(String(32), nullable=False)
    action_type = Column(String(64), nullable=False)
    actor = Column(String(64), default="RecoverAI_Agent")
    idempotency_key = Column(String(128), nullable=True)
    cost_incurred_inr = Column(Float, default=0.0)
    amount_recovered_inr = Column(Float, default=0.0)
    details_json = Column(JSON, default=dict)

    transaction = relationship("TransactionDB", back_populates="audit_events")
