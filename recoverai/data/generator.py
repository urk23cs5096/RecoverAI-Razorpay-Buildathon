"""
Realistic Synthetic Transaction & Payment Failure Generator for Indian Digital Commerce on Razorpay.

Generates multi-merchant, multi-channel payment streams with authentic error distributions,
temporal bank maintenance patterns, customer tier dynamics, and ground-truth recovery labels.
"""

import random
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Tuple
import numpy as np
import pandas as pd

from recoverai.data.schemas import (
    PaymentMethod,
    FailureCategory,
    CustomerTier,
    CommunicationChannel,
    InterventionType,
)
from recoverai.config.settings import settings

logger = logging.getLogger("recoverai.data.generator")

# Set deterministic seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

MERCHANTS = [
    {
        "merchant_id": "merch_saas_metrics",
        "merchant_name": "MetricsCloud SaaS",
        "merchant_category": "SaaS_B2B",
        "base_amount_range": (2499.0, 39999.0),
        "recurring_prob": 0.85,
        "payment_method_weights": [0.15, 0.50, 0.10, 0.20, 0.05],  # UPI, CARD, NETBANKING, EMANDATE_CARD, EMANDATE_UPI
    },
    {
        "merchant_id": "merch_urban_threads",
        "merchant_name": "UrbanThreads D2C",
        "merchant_category": "D2C_Ecommerce",
        "base_amount_range": (499.0, 4999.0),
        "recurring_prob": 0.05,
        "payment_method_weights": [0.65, 0.20, 0.05, 0.02, 0.08],
    },
    {
        "merchant_id": "merch_learn_hub",
        "merchant_name": "LearnHub EdTech",
        "merchant_category": "EdTech_Subscription",
        "base_amount_range": (999.0, 14999.0),
        "recurring_prob": 0.60,
        "payment_method_weights": [0.35, 0.25, 0.10, 0.10, 0.20],
    },
    {
        "merchant_id": "merch_stream_now",
        "merchant_name": "StreamNow OTT",
        "merchant_category": "OTT_Media",
        "base_amount_range": (199.0, 1199.0),
        "recurring_prob": 0.95,
        "payment_method_weights": [0.20, 0.25, 0.05, 0.25, 0.25],
    },
    {
        "merchant_id": "merch_enterprise_erp",
        "merchant_name": "Apex Enterprise Solutions",
        "merchant_category": "B2B_Invoicing",
        "base_amount_range": (35000.0, 125000.0),
        "recurring_prob": 0.30,
        "payment_method_weights": [0.05, 0.40, 0.45, 0.08, 0.02],
    },
]

BANKS = ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "YESB", "PNB", "FEDERAL"]
BANK_WEIGHTS = [0.28, 0.24, 0.20, 0.12, 0.08, 0.04, 0.02, 0.02]

CARD_NETWORKS = ["VISA", "MASTERCARD", "RUPAY", "AMEX"]
CARD_WEIGHTS = [0.45, 0.35, 0.15, 0.05]

FAILURE_REASONS_CATALOG = {
    FailureCategory.BANK_DOWNTIME: [
        ("U19", "NPCI / Beneficiary bank server unresponsive"),
        ("91", "Card Issuer switch down or timeout"),
        ("GATEWAY_TIMEOUT", "Payment gateway connection timed out during bank routing"),
    ],
    FailureCategory.INSUFFICIENT_FUNDS: [
        ("51", "Insufficient funds in customer account / credit limit exhausted"),
        ("U66", "UPI transaction declined due to insufficient balance"),
        ("BALANCE_LOW", "Pre-debit balance check failed"),
    ],
    FailureCategory.AUTH_FAILED_3DS: [
        ("3DS_TIMEOUT", "Customer failed to complete 3DS OTP authentication in time"),
        ("OTP_INCORRECT", "Customer entered invalid OTP multiple times"),
        ("USER_DROPPED", "Customer aborted 3DS authentication window"),
    ],
    FailureCategory.MANDATE_EXPIRED: [
        ("U30", "e-Mandate registration expired or revoked by customer"),
        ("MANDATE_INVALID", "Card token on recurring mandate expired"),
        ("MANDATE_LIMIT", "Recurring debit amount exceeds mandate threshold"),
    ],
    FailureCategory.CARD_LIMIT_EXCEEDED: [
        ("61", "Transaction amount exceeds online e-commerce card daily limit"),
        ("TXN_LIMIT_EXCEEDED", "Exceeded per-transaction banking limit"),
    ],
    FailureCategory.NETWORK_ERROR: [
        ("NET_DROP", "Transient network drop between client browser and gateway"),
        ("SOCKET_TIMEOUT", "Socket timeout during cryptographic handshake"),
    ],
}


class SyntheticDataGenerator:
    """Generates realistic synthetic transaction streams modeling Indian payment rails."""

    def __init__(self, seed: int = RANDOM_SEED):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def generate_dataset(
        self,
        num_transactions: int = 10000,
        start_date: datetime = datetime(2026, 1, 1),
        days_span: int = 60,
    ) -> pd.DataFrame:
        """
        Generates a synthetic dataset of payment failures and outcomes.
        """
        logger.info(f"Generating {num_transactions} synthetic payment failure records...")
        records: List[Dict[str, Any]] = []

        # Pre-generate a fixed customer base to establish realistic history & LTV
        num_customers = int(num_transactions * 0.35)
        customer_pool = self._generate_customer_pool(num_customers)

        for i in range(num_transactions):
            merchant = random.choice(MERCHANTS)
            customer = random.choice(customer_pool)
            
            # Timestamp distribution with peak maintenance hours (23:00 - 03:00)
            txn_time = self._generate_timestamp(start_date, days_span)
            hour = txn_time.hour
            day_of_week = txn_time.weekday()
            day_of_month = txn_time.day

            # Payment method selection
            payment_methods = list(PaymentMethod)
            payment_method = random.choices(
                payment_methods, weights=merchant["payment_method_weights"], k=1
            )[0]

            # Bank selection
            bank_code = random.choices(BANKS, weights=BANK_WEIGHTS, k=1)[0]
            card_network = (
                random.choices(CARD_NETWORKS, weights=CARD_WEIGHTS, k=1)[0]
                if "CARD" in payment_method.value
                else "NONE"
            )

            is_recurring = (
                random.random() < merchant["recurring_prob"]
                or "EMANDATE" in payment_method.value
            )
            mandate_id = f"man_{random.randint(100000, 999999)}" if is_recurring else None

            # Base amount with log-normal distribution inside merchant range
            low, high = merchant["base_amount_range"]
            amount_inr = round(np.random.uniform(low, high), 2)

            # Determine Failure Category based on realistic conditional probabilities
            failure_category = self._sample_failure_category(
                hour=hour,
                day_of_month=day_of_month,
                is_recurring=is_recurring,
                payment_method=payment_method,
                amount_inr=amount_inr,
            )

            # Pick gateway error code
            error_code_pair = random.choice(FAILURE_REASONS_CATALOG[failure_category])
            gateway_error_code, gateway_error_desc = error_code_pair

            retry_count = 0
            if random.random() < 0.25:
                retry_count = random.randint(1, 2)

            # Ground truth simulation: What is the optimal recovery action and will it succeed?
            (
                optimal_action,
                optimal_delay_hours,
                ground_truth_recovery_prob,
                is_recovered_under_optimal,
                is_recovered_under_naive_immediate,
            ) = self._compute_ground_truth_outcomes(
                failure_category=failure_category,
                bank_code=bank_code,
                customer_tier=customer["customer_tier"],
                amount_inr=amount_inr,
                payment_method=payment_method,
                hour=hour,
                day_of_month=day_of_month,
            )

            record = {
                "transaction_id": f"txn_{100000 + i}",
                "order_id": f"order_{200000 + i}",
                "merchant_id": merchant["merchant_id"],
                "merchant_category": merchant["merchant_category"],
                "customer_id": customer["customer_id"],
                "customer_tier": customer["customer_tier"],
                "customer_ltv_inr": customer["customer_ltv_inr"],
                "preferred_channel": customer["preferred_channel"],
                "amount_inr": amount_inr,
                "currency": "INR",
                "payment_method": payment_method.value,
                "bank_code": bank_code,
                "card_network": card_network,
                "is_recurring": is_recurring,
                "mandate_id": mandate_id,
                "timestamp": txn_time.strftime("%Y-%m-%d %H:%M:%S"),
                "hour_of_day": hour,
                "day_of_week": day_of_week,
                "day_of_month": day_of_month,
                "is_failed": True,
                "gateway_error_code": gateway_error_code,
                "gateway_error_description": gateway_error_desc,
                "failure_category": failure_category.value,
                "retry_attempt_count": retry_count,
                # Ground truth target labels
                "ground_truth_optimal_action": optimal_action.value,
                "ground_truth_optimal_delay_hours": optimal_delay_hours,
                "ground_truth_recovery_prob": round(ground_truth_recovery_prob, 4),
                "recovered_optimal": int(is_recovered_under_optimal),
                "recovered_naive_immediate": int(is_recovered_under_naive_immediate),
            }
            records.append(record)

        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(by="timestamp").reset_index(drop=True)
        logger.info(f"Generated dataset shape: {df.shape}")
        return df

    def _generate_customer_pool(self, count: int) -> List[Dict[str, Any]]:
        """Generates pool of customers with consistent tiers and LTVs."""
        tiers = [CustomerTier.VIP, CustomerTier.REGULAR, CustomerTier.NEW, CustomerTier.AT_RISK]
        tier_weights = [0.10, 0.60, 0.20, 0.10]
        channels = [CommunicationChannel.WHATSAPP, CommunicationChannel.SMS, CommunicationChannel.EMAIL]
        channel_weights = [0.70, 0.20, 0.10]

        pool = []
        for i in range(count):
            tier = random.choices(tiers, weights=tier_weights, k=1)[0]
            ltv = {
                CustomerTier.VIP: np.random.uniform(50000.0, 300000.0),
                CustomerTier.REGULAR: np.random.uniform(5000.0, 50000.0),
                CustomerTier.NEW: np.random.uniform(500.0, 5000.0),
                CustomerTier.AT_RISK: np.random.uniform(1000.0, 15000.0),
            }[tier]
            
            chosen_channel = random.choices(channels, weights=channel_weights, k=1)[0]
            pool.append({
                "customer_id": f"cust_{50000 + i}",
                "customer_tier": tier.value,
                "customer_ltv_inr": round(float(ltv), 2),
                "preferred_channel": chosen_channel.value,
            })
        return pool

    def _generate_timestamp(self, start_date: datetime, days_span: int) -> datetime:
        """Generates realistic timestamps with higher failure density during bank maintenance hours."""
        random_day = random.randint(0, days_span - 1)
        # 30% of failures happen during maintenance hours (23:00 - 03:00)
        if random.random() < 0.30:
            hour = random.choice([23, 0, 1, 2, 3])
        else:
            hour = random.randint(4, 22)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        return start_date + timedelta(days=random_day, hours=hour, minutes=minute, seconds=second)

    def _sample_failure_category(
        self,
        hour: int,
        day_of_month: int,
        is_recurring: bool,
        payment_method: PaymentMethod,
        amount_inr: float,
    ) -> FailureCategory:
        """Samples failure category with domain-realistic priors."""
        # High bank downtime during midnight maintenance
        if hour in [23, 0, 1, 2, 3] and random.random() < 0.65:
            return FailureCategory.BANK_DOWNTIME
        
        # High insufficient funds near month-end (23-29)
        if day_of_month in range(23, 30) and random.random() < 0.40:
            return FailureCategory.INSUFFICIENT_FUNDS

        # Mandate expired on recurring payments
        if is_recurring and random.random() < 0.45:
            return FailureCategory.MANDATE_EXPIRED

        # Card limit exceeded for high amounts
        if amount_inr > 25000.0 and "CARD" in payment_method.value and random.random() < 0.40:
            return FailureCategory.CARD_LIMIT_EXCEEDED

        # 3DS authentication failure for non-recurring online checkouts
        if not is_recurring and ("CARD" in payment_method.value or payment_method == PaymentMethod.NETBANKING):
            if random.random() < 0.50:
                return FailureCategory.AUTH_FAILED_3DS

        # Default distribution across remaining categories
        weights = [0.20, 0.25, 0.25, 0.10, 0.10, 0.10]
        categories = list(FailureCategory)
        return random.choices(categories, weights=weights, k=1)[0]

    def _compute_ground_truth_outcomes(
        self,
        failure_category: FailureCategory,
        bank_code: str,
        customer_tier: str,
        amount_inr: float,
        payment_method: PaymentMethod,
        hour: int,
        day_of_month: int,
    ) -> Tuple[InterventionType, int, float, bool, bool]:
        """
        Calculates ground truth optimal intervention, optimal delay, expected recovery probability,
        and counterfactual success under Optimal vs. Naive Immediate Retries.
        """
        # Bank reliability adjustment
        bank_boost = 0.05 if bank_code in ["HDFC", "ICICI", "KOTAK"] else -0.05
        vip_boost = 0.10 if customer_tier == CustomerTier.VIP.value else 0.0

        if failure_category == FailureCategory.BANK_DOWNTIME:
            # Bank downtime recovers best with delayed smart retry after switch restores
            optimal_action = InterventionType.SMART_RETRY_DELAYED
            optimal_delay_hours = 2 if hour in [23, 0, 1, 2] else 1
            base_prob = 0.88 + bank_boost
            naive_prob = 0.08  # Immediate retry while bank is down fails 92% of the time

        elif failure_category == FailureCategory.INSUFFICIENT_FUNDS:
            # Insufficient funds recovers best with delayed WhatsApp pay link or waiting for salary
            optimal_action = InterventionType.WHATSAPP_PAY_LINK
            optimal_delay_hours = 24 if day_of_month >= 28 else 48
            base_prob = 0.65 + vip_boost
            naive_prob = 0.05  # Immediate retry fails because funds haven't arrived

        elif failure_category == FailureCategory.AUTH_FAILED_3DS:
            # 3DS drop recovers best with 1-Click Pay Link or UPI Intent switch immediately
            if payment_method == PaymentMethod.CARD:
                optimal_action = InterventionType.UPI_INTENT_SWITCH
            else:
                optimal_action = InterventionType.WHATSAPP_PAY_LINK
            optimal_delay_hours = 0
            base_prob = 0.78 + vip_boost
            naive_prob = 0.00  # Auto-retrying a card without customer entering OTP impossible (0%)

        elif failure_category == FailureCategory.MANDATE_EXPIRED:
            # Mandate expired requires mandate re-auth link via WhatsApp/SMS
            optimal_action = InterventionType.WHATSAPP_PAY_LINK
            optimal_delay_hours = 0
            base_prob = 0.72 + vip_boost
            naive_prob = 0.00  # Auto-retrying an expired mandate is rejected by NPCI/Issuer (0%)

        elif failure_category == FailureCategory.CARD_LIMIT_EXCEEDED:
            # High amount card limit exceeded -> Switch payment method to NetBanking/UPI or escalate
            if amount_inr >= settings.HIGH_VALUE_THRESHOLD_INR:
                optimal_action = InterventionType.MANUAL_ESCALATION
            else:
                optimal_action = InterventionType.UPI_INTENT_SWITCH
            optimal_delay_hours = 0
            base_prob = 0.70 + vip_boost
            naive_prob = 0.02  # Retrying the same card still hits the limit

        else:  # NETWORK_ERROR
            optimal_action = InterventionType.SMART_RETRY_IMMEDIATE
            optimal_delay_hours = 0
            base_prob = 0.85
            naive_prob = 0.75  # Immediate retry for transient network drop has decent recovery

        # Clamp probabilities
        optimal_prob = min(max(base_prob, 0.05), 0.96)
        naive_prob = min(max(naive_prob, 0.00), 0.90)

        # Stochastic outcome draws
        recovered_optimal = random.random() < optimal_prob
        recovered_naive = random.random() < naive_prob

        return optimal_action, optimal_delay_hours, optimal_prob, recovered_optimal, recovered_naive


def generate_and_save_data(output_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Generates synthetic dataset, validates it, and saves train/val/test splits."""
    generator = SyntheticDataGenerator()
    df = generator.generate_dataset(num_transactions=10000)

    # Validate dataset
    from recoverai.data.validator import DataContractValidator
    cleaned_df, report = DataContractValidator.validate_dataframe(df)
    logger.info(f"Dataset Validation Report: {report}")

    # Chronological Split (70% Train, 15% Validation, 15% Test) to prevent temporal data leakage
    n = len(cleaned_df)
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.85)

    train_df = cleaned_df.iloc[:train_idx].copy()
    val_df = cleaned_df.iloc[train_idx:val_idx].copy()
    test_df = cleaned_df.iloc[val_idx:].copy()

    settings.ensure_directories()
    raw_path = settings.DATA_DIR / "raw" / "transactions_synthetic.csv"
    train_path = settings.DATA_DIR / "processed" / "train.csv"
    val_path = settings.DATA_DIR / "processed" / "val.csv"
    test_path = settings.DATA_DIR / "processed" / "test.csv"

    cleaned_df.to_csv(raw_path, index=False)
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"Saved: {raw_path} ({len(cleaned_df)} rows)")
    logger.info(f"Saved: {train_path} ({len(train_df)} rows), {val_path} ({len(val_df)} rows), {test_path} ({len(test_df)} rows)")

    return train_df, val_df, test_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    generate_and_save_data()
