"""Interactive Live Agent Recovery Workbench."""

from datetime import datetime, timezone
import streamlit as st
import json

from recoverai.data.schemas import (
    PaymentTransaction,
    PaymentMethod,
    FailureCategory,
    CustomerTier,
    CommunicationChannel,
    InterventionType,
)
from recoverai.agent.workflow import RecoveryAgentWorkflow
from recoverai.simulator.engine import RecoverySimulatorEngine
from recoverai.storage.database import StorageRepository
from recoverai.storage.audit import AuditLogger
from recoverai.config.settings import settings

PRESETS = {
    "1. HDFC Bank Downtime (U19 @ 01:30 AM)": {
        "transaction_id": "txn_demo_bank_dt",
        "merchant_id": "merch_saas_metrics",
        "merchant_category": "SaaS_B2B",
        "customer_id": "cust_vip_401",
        "customer_tier": CustomerTier.VIP,
        "customer_ltv_inr": 120000.0,
        "preferred_channel": CommunicationChannel.WHATSAPP,
        "amount_inr": 4999.0,
        "payment_method": PaymentMethod.CARD,
        "bank_code": "HDFC",
        "card_network": "VISA",
        "is_recurring": True,
        "hour_of_day": 1,
        "day_of_week": 2,
        "gateway_error_code": "U19",
        "gateway_error_description": "Beneficiary bank server unresponsive",
        "failure_category": FailureCategory.BANK_DOWNTIME,
        "retry_attempt_count": 0,
    },
    "2. D2C Checkout 3DS Dropped (3DS_TIMEOUT)": {
        "transaction_id": "txn_demo_3ds_drop",
        "merchant_id": "merch_urban_threads",
        "merchant_category": "D2C_Ecommerce",
        "customer_id": "cust_reg_102",
        "customer_tier": CustomerTier.REGULAR,
        "customer_ltv_inr": 8500.0,
        "preferred_channel": CommunicationChannel.WHATSAPP,
        "amount_inr": 1899.0,
        "payment_method": PaymentMethod.CARD,
        "bank_code": "ICICI",
        "card_network": "MASTERCARD",
        "is_recurring": False,
        "hour_of_day": 15,
        "day_of_week": 5,
        "gateway_error_code": "3DS_TIMEOUT",
        "gateway_error_description": "Customer aborted OTP verification window",
        "failure_category": FailureCategory.AUTH_FAILED_3DS,
        "retry_attempt_count": 0,
    },
    "3. High-Value B2B Invoice (₹75,000 Circuit Breaker)": {
        "transaction_id": "txn_demo_high_val",
        "merchant_id": "merch_enterprise_erp",
        "merchant_category": "B2B_Invoicing",
        "customer_id": "cust_ent_909",
        "customer_tier": CustomerTier.VIP,
        "customer_ltv_inr": 450000.0,
        "preferred_channel": CommunicationChannel.EMAIL,
        "amount_inr": 75000.0,
        "payment_method": PaymentMethod.NETBANKING,
        "bank_code": "SBI",
        "card_network": "NONE",
        "is_recurring": False,
        "hour_of_day": 11,
        "day_of_week": 1,
        "gateway_error_code": "61",
        "gateway_error_description": "Transaction amount exceeds online limit",
        "failure_category": FailureCategory.CARD_LIMIT_EXCEEDED,
        "retry_attempt_count": 0,
    },
    "4. Expired Mandate Involuntary Churn (U30)": {
        "transaction_id": "txn_demo_mandate_exp",
        "merchant_id": "merch_learn_hub",
        "merchant_category": "EdTech_Subscription",
        "customer_id": "cust_sub_332",
        "customer_tier": CustomerTier.REGULAR,
        "customer_ltv_inr": 25000.0,
        "preferred_channel": CommunicationChannel.WHATSAPP,
        "amount_inr": 2499.0,
        "payment_method": PaymentMethod.EMANDATE_CARD,
        "bank_code": "KOTAK",
        "card_network": "VISA",
        "is_recurring": True,
        "hour_of_day": 9,
        "day_of_week": 4,
        "gateway_error_code": "U30",
        "gateway_error_description": "e-Mandate registration expired or token invalid",
        "failure_category": FailureCategory.MANDATE_EXPIRED,
        "retry_attempt_count": 0,
    },
}


def render_workbench_view():
    st.markdown("## ⚡ Live Autonomous Recovery Agent Workbench")
    st.caption("Interact with the full agent loop in real-time: Observe ML scoring, Expected Value decisioning, LLM diagnosis, guardrail circuit breakers, and simulated outcomes.")

    col_cfg, col_exec = st.columns([1, 2])

    with col_cfg:
        st.markdown("### 🛠 Incident Configuration")
        preset_choice = st.selectbox("Select Failure Scenario Preset", list(PRESETS.keys()) + ["Custom Manual Setup"])

        if preset_choice != "Custom Manual Setup":
            preset_data = PRESETS[preset_choice]
            txn_id = preset_data["transaction_id"]
            merchant_id = preset_data["merchant_id"]
            merchant_cat = preset_data["merchant_category"]
            customer_id = preset_data["customer_id"]
            tier = preset_data["customer_tier"]
            ltv = preset_data["customer_ltv_inr"]
            channel = preset_data["preferred_channel"]
            amount = preset_data["amount_inr"]
            method = preset_data["payment_method"]
            bank = preset_data["bank_code"]
            network = preset_data["card_network"]
            is_rec = preset_data["is_recurring"]
            hour = preset_data["hour_of_day"]
            error_code = preset_data["gateway_error_code"]
            error_desc = preset_data["gateway_error_description"]
            cat = preset_data["failure_category"]
            retries = preset_data["retry_attempt_count"]
        else:
            txn_id = f"txn_custom_{int(datetime.now(timezone.utc).timestamp())}"
            merchant_id = "merch_saas_metrics"
            merchant_cat = "SaaS_B2B"
            customer_id = "cust_custom_01"
            tier = st.selectbox("Customer Tier", list(CustomerTier))
            ltv = st.number_input("Customer LTV (₹)", value=25000.0, step=1000.0)
            channel = st.selectbox("Preferred Channel", list(CommunicationChannel))
            amount = st.number_input("Transaction Amount (₹)", value=3499.0, step=100.0)
            method = st.selectbox("Payment Method", list(PaymentMethod))
            bank = st.selectbox("Bank", ["HDFC", "ICICI", "SBI", "AXIS", "KOTAK", "YESB"])
            network = "VISA" if "CARD" in method.value else "NONE"
            is_rec = st.checkbox("Is Recurring Subscription", value=True)
            hour = st.slider("Hour of Day (0-23)", 0, 23, 14)
            cat = st.selectbox("Failure Category", list(FailureCategory))
            error_code = "U19" if cat == FailureCategory.BANK_DOWNTIME else "51"
            error_desc = "Gateway failure description"
            retries = st.number_input("Past Retry Count", min_value=0, max_value=5, value=0)

        # Summary box
        st.info(
            f"**Incident Summary:**\n"
            f"- **Txn ID:** `{txn_id}`\n"
            f"- **Amount:** ₹{amount:,.2f} | **Bank:** {bank}\n"
            f"- **Category:** `{cat.value if hasattr(cat, 'value') else cat}` ({error_code})\n"
            f"- **Customer:** `{customer_id}` ({tier.value if hasattr(tier, 'value') else tier}, LTV: ₹{ltv:,.0f})"
        )

        run_btn = st.button("🚀 Trigger RecoverAI Agent Loop", type="primary", use_container_width=True)

    with col_exec:
        st.markdown("### 🔍 Agent Execution Trace & Intelligence")

        if run_btn:
            txn_obj = PaymentTransaction(
                transaction_id=txn_id,
                merchant_id=merchant_id,
                merchant_category=merchant_cat,
                customer_id=customer_id,
                customer_tier=tier,
                customer_ltv_inr=ltv,
                preferred_channel=channel,
                amount_inr=amount,
                currency="INR",
                payment_method=method,
                bank_code=bank,
                card_network=network,
                is_recurring=is_rec,
                timestamp=datetime.now(timezone.utc).replace(tzinfo=None),
                hour_of_day=hour,
                day_of_week=3,
                gateway_error_code=error_code,
                gateway_error_description=error_desc,
                failure_category=cat,
                retry_attempt_count=retries,
            )

            workflow = RecoveryAgentWorkflow()
            simulator = RecoverySimulatorEngine()
            repo = StorageRepository()
            audit_logger = AuditLogger(repo)

            # Persist txn in DB
            repo.upsert_transaction(txn_obj.model_dump())

            with st.spinner("Agent running: Detect ➔ Diagnose ➔ Decide ➔ Guardrail ➔ Act ➔ Verify..."):
                agent_res = workflow.process_incident(txn_obj)
                decision = agent_res["decision"]
                explanation = agent_res["explanation"]

                sim_outcome = None
                if agent_res["action_executed"]:
                    sim_outcome = simulator.simulate_outcome(
                        txn=txn_obj,
                        action=decision.recommended_action,
                        delay_hours=decision.recommended_delay_hours,
                    )
                    final_state = "RECOVERED" if sim_outcome["is_recovered"] else "FAILED_TERMINAL"
                    cost = sim_outcome["operational_cost_inr"]
                    recov_amt = sim_outcome["amount_recovered_inr"] if sim_outcome["is_recovered"] else 0.0

                    audit_logger.log_transition(
                        transaction_id=txn_id,
                        previous_state=agent_res["final_state"],
                        current_state=final_state,
                        action_type=decision.recommended_action,
                        idempotency_key=agent_res.get("idempotency_key"),
                        cost_incurred_inr=cost,
                        amount_recovered_inr=recov_amt,
                        details_json=sim_outcome,
                    )
                    repo.upsert_transaction({
                        "transaction_id": txn_id,
                        "merchant_id": merchant_id,
                        "customer_id": customer_id,
                        "amount_inr": amount,
                        "payment_method": method.value if hasattr(method, "value") else method,
                        "bank_code": bank,
                        "failure_category": cat.value if hasattr(cat, "value") else cat,
                        "gateway_error_code": error_code,
                        "current_agent_state": final_state,
                        "is_recovered": sim_outcome["is_recovered"],
                        "amount_recovered_inr": recov_amt,
                        "retry_attempt_count": retries + 1,
                    })

            # Display Timeline Traces
            t1, t2, t3 = st.tabs(["1. Decision & Expected Value", "2. LLM Reasoning & Copy", "3. Guardrails & Simulation Outcome"])

            with t1:
                st.markdown("#### 📐 Mathematical Expected Value Optimization")
                st.write(f"**Recommended Action:** `{decision.recommended_action.value}`")
                st.write(f"**Recommended Backoff:** `{decision.recommended_delay_hours} hours`")
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Recovery Probability (P)", f"{decision.estimated_recovery_probability * 100:.1f}%")
                m2.metric("Intervention Cost", f"₹{decision.intervention_cost_inr:.2f}")
                m3.metric("Net Expected ROI", f"₹{decision.net_expected_roi_inr:,.2f}")

                st.success(f"**Decision Summary:** {decision.reasoning_summary}")
                if decision.safety_rule_applied:
                    st.warning(f"**Safety Rule Override:** {decision.safety_rule_applied}")

            with t2:
                st.markdown("#### 🤖 LLM Contextual Reasoning & Communication")
                st.write("**Root Cause Diagnosis:**")
                st.info(explanation.root_cause_diagnosis)

                st.write("**Merchant Operations Guidance:**")
                st.write(explanation.merchant_recommendation)

                st.write(f"**Drafted Customer Communication ({explanation.channel_selected.value if hasattr(explanation.channel_selected, 'value') else explanation.channel_selected}):**")
                st.code(explanation.customer_recovery_message, language="markdown")

            with t3:
                st.markdown("#### 🛡️ Guardrails & Physical Outcome Verification")
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    st.write(f"**Guardrail Status:** `{agent_res['guardrail_status']}`")
                    st.write(f"**Idempotency Key:** `{agent_res.get('idempotency_key', 'N/A')[:16]}...`" if agent_res.get("idempotency_key") else "**Idempotency Key:** None")
                    st.write(f"**Action Executed:** `{agent_res['action_executed']}`")
                
                with col_g2:
                    if sim_outcome:
                        status_color = "green" if sim_outcome["is_recovered"] else "red"
                        status_text = "SUCCESSFULLY RECOVERED" if sim_outcome["is_recovered"] else "FAILED TO RECOVER"
                        st.markdown(f"**Simulated Outcome:** :{status_color}[**{status_text}**]")
                        st.write(f"**Amount Recovered:** ₹{sim_outcome['amount_recovered_inr']:,.2f}")
                        st.write(f"**Time to Recovery:** {sim_outcome['time_to_recovery_hours']:.1f} hours")
                        st.write(f"**Details:** {sim_outcome['outcome_description']}")
                    else:
                        st.warning(f"Action suppressed or escalated: {agent_res.get('guardrail_reason', 'N/A')}")
        else:
            st.info("👈 Select a preset scenario on the left and click **Trigger RecoverAI Agent Loop** to observe the execution trace.")
