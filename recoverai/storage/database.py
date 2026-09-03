"""Database session management and repository queries."""

import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from sqlalchemy import create_engine, func, desc
from sqlalchemy.orm import sessionmaker, Session

from recoverai.storage.models import Base, TransactionDB, DecisionDB, AuditEventDB
from recoverai.config.settings import settings

logger = logging.getLogger("recoverai.storage.database")

engine = create_engine(
    settings.DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in settings.DATABASE_URL else {},
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initializes database schema."""
    Base.metadata.create_all(bind=engine)
    logger.info("Database schema initialized.")


def get_db():
    """Dependency generator for database sessions."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


class StorageRepository:
    """Repository helper for transaction, decision, and audit persistence."""

    def __init__(self, db_session: Optional[Session] = None):
        self._db = db_session

    def get_session(self) -> Session:
        return self._db if self._db is not None else SessionLocal()

    def upsert_transaction(self, txn_dict: Dict[str, Any]) -> TransactionDB:
        session = self.get_session()
        try:
            txn_id = txn_dict["transaction_id"]
            existing = session.query(TransactionDB).filter(TransactionDB.transaction_id == txn_id).first()
            if not existing:
                existing = TransactionDB(
                    transaction_id=txn_id,
                    order_id=txn_dict.get("order_id"),
                    merchant_id=txn_dict["merchant_id"],
                    merchant_category=txn_dict.get("merchant_category", "General"),
                    customer_id=txn_dict["customer_id"],
                    customer_tier=txn_dict.get("customer_tier", "REGULAR"),
                    customer_ltv_inr=float(txn_dict.get("customer_ltv_inr", 0.0)),
                    preferred_channel=txn_dict.get("preferred_channel", "WHATSAPP"),
                    amount_inr=float(txn_dict["amount_inr"]),
                    currency=txn_dict.get("currency", "INR"),
                    payment_method=txn_dict["payment_method"],
                    bank_code=txn_dict["bank_code"],
                    card_network=txn_dict.get("card_network", "NONE"),
                    is_recurring=bool(txn_dict.get("is_recurring", False)),
                    mandate_id=txn_dict.get("mandate_id"),
                    failure_category=txn_dict["failure_category"],
                    gateway_error_code=txn_dict["gateway_error_code"],
                    gateway_error_description=txn_dict.get("gateway_error_description", ""),
                    retry_attempt_count=int(txn_dict.get("retry_attempt_count", 0)),
                    current_agent_state=txn_dict.get("current_agent_state", "DETECTED"),
                    is_recovered=bool(txn_dict.get("is_recovered", False)),
                    amount_recovered_inr=float(txn_dict.get("amount_recovered_inr", 0.0)),
                )
                session.add(existing)
            else:
                existing.retry_attempt_count = int(txn_dict.get("retry_attempt_count", existing.retry_attempt_count))
                existing.current_agent_state = txn_dict.get("current_agent_state", existing.current_agent_state)
                existing.is_recovered = bool(txn_dict.get("is_recovered", existing.is_recovered))
                existing.amount_recovered_inr = float(txn_dict.get("amount_recovered_inr", existing.amount_recovered_inr))
                existing.updated_at = datetime.now(timezone.utc).replace(tzinfo=None)
            session.commit()
            session.refresh(existing)
            return existing
        finally:
            if self._db is None:
                session.close()

    def record_decision(self, decision_dict: Dict[str, Any]) -> DecisionDB:
        session = self.get_session()
        try:
            rec = DecisionDB(**decision_dict)
            session.add(rec)
            session.commit()
            session.refresh(rec)
            return rec
        finally:
            if self._db is None:
                session.close()

    def record_audit_event(self, event_dict: Dict[str, Any]) -> AuditEventDB:
        session = self.get_session()
        try:
            rec = AuditEventDB(**event_dict)
            session.add(rec)
            session.commit()
            session.refresh(rec)
            return rec
        finally:
            if self._db is None:
                session.close()

    def list_transactions(self, limit: int = 100, merchant_id: Optional[str] = None) -> List[TransactionDB]:
        session = self.get_session()
        try:
            q = session.query(TransactionDB)
            if merchant_id:
                q = q.filter(TransactionDB.merchant_id == merchant_id)
            return q.order_by(desc(TransactionDB.created_at)).limit(limit).all()
        finally:
            if self._db is None:
                session.close()

    def get_transaction_by_id(self, txn_id: str) -> Optional[TransactionDB]:
        session = self.get_session()
        try:
            return session.query(TransactionDB).filter(TransactionDB.transaction_id == txn_id).first()
        finally:
            if self._db is None:
                session.close()

    def get_audit_trail_for_txn(self, txn_id: str) -> List[AuditEventDB]:
        session = self.get_session()
        try:
            return (
                session.query(AuditEventDB)
                .filter(AuditEventDB.transaction_id == txn_id)
                .order_by(AuditEventDB.timestamp.asc())
                .all()
            )
        finally:
            if self._db is None:
                session.close()

    def get_executive_summary_stats(self) -> Dict[str, Any]:
        session = self.get_session()
        try:
            total_txns = session.query(func.count(TransactionDB.id)).scalar() or 0
            total_at_risk = session.query(func.sum(TransactionDB.amount_inr)).scalar() or 0.0
            total_recovered = session.query(func.sum(TransactionDB.amount_recovered_inr)).scalar() or 0.0
            recovered_count = session.query(func.count(TransactionDB.id)).filter(TransactionDB.is_recovered == True).scalar() or 0
            
            total_costs = session.query(func.sum(AuditEventDB.cost_incurred_inr)).scalar() or 0.0
            
            recovery_rate = (total_recovered / total_at_risk * 100.0) if total_at_risk > 0 else 0.0
            net_recovered = total_recovered - total_costs

            return {
                "total_failed_incidents": total_txns,
                "total_revenue_at_risk_inr": round(float(total_at_risk), 2),
                "gross_revenue_recovered_inr": round(float(total_recovered), 2),
                "total_operational_costs_inr": round(float(total_costs), 2),
                "net_revenue_recovered_inr": round(float(net_recovered), 2),
                "recovered_transaction_count": recovered_count,
                "recovery_rate_pct": round(recovery_rate, 2),
            }
        finally:
            if self._db is None:
                session.close()
