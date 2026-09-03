"""Audit Trail and Transaction Drilldown View."""

import streamlit as st
import pandas as pd
from recoverai.storage.database import StorageRepository


def render_audit_view():
    st.markdown("## 📜 Immutable Audit Trail & Incident Drilldown")
    st.caption("Inspect persisted state transitions, cryptographic idempotency hashes, and financial accounting logs.")

    repo = StorageRepository()
    txns = repo.list_transactions(limit=100)

    if not txns:
        st.info("No live transactions processed yet. Run scenarios in the **Live Agent Workbench** or trigger batch simulation.")
        return

    txn_data = [
        {
            "Transaction ID": t.transaction_id,
            "Merchant": t.merchant_id,
            "Customer ID": t.customer_id,
            "Amount (₹)": f"₹{t.amount_inr:,.2f}",
            "Method": t.payment_method,
            "Bank": t.bank_code,
            "Failure Category": t.failure_category,
            "Error Code": t.gateway_error_code,
            "Agent State": t.current_agent_state,
            "Recovered": "✅ Yes" if t.is_recovered else "❌ No",
            "Recovered (₹)": f"₹{t.amount_recovered_inr:,.2f}",
            "Created At": t.created_at.strftime("%Y-%m-%d %H:%M:%S") if t.created_at else "N/A",
        }
        for t in txns
    ]

    df_table = pd.DataFrame(txn_data)
    st.dataframe(df_table, use_container_width=True, height=300)

    st.markdown("---")
    st.markdown("### 🔍 Per-Transaction Audit Deep Dive")

    txn_ids = [t.transaction_id for t in txns]
    selected_txn_id = st.selectbox("Select Transaction to Inspect Audit Trail", txn_ids)

    if selected_txn_id:
        audits = repo.get_audit_trail_for_txn(selected_txn_id)
        txn_rec = repo.get_transaction_by_id(selected_txn_id)

        if txn_rec:
            col_d1, col_d2, col_d3 = st.columns(3)
            col_d1.metric("Transaction Amount", f"₹{txn_rec.amount_inr:,.2f}")
            col_d2.metric("Final Agent State", txn_rec.current_agent_state)
            col_d3.metric("Gross Recovered", f"₹{txn_rec.amount_recovered_inr:,.2f}")

        if audits:
            st.markdown("##### Lifecycle Transition Events")
            audit_events = [
                {
                    "Timestamp (UTC)": a.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "Transition": f"{a.previous_state} ➔ {a.current_state}",
                    "Action Dispatched": a.action_type,
                    "Actor": a.actor,
                    "Idempotency Key": a.idempotency_key[:16] + "..." if a.idempotency_key else "None",
                    "Cost Incurred (₹)": f"₹{a.cost_incurred_inr:.2f}",
                    "Amount Recovered (₹)": f"₹{a.amount_recovered_inr:,.2f}",
                }
                for a in audits
            ]
            st.table(pd.DataFrame(audit_events))

            # Display Decision Provenance if details_json exists
            for a in audits:
                if a.details_json and any(k in a.details_json for k in ["provenance", "calibrated_probability", "decision_ev_inr", "guardrails_passed"]):
                    details = a.details_json
                    with st.expander(f"🔬 Decision Provenance & Cryptographic Audit Proof ({a.current_state})", expanded=True):
                        col_p1, col_p2, col_p3 = st.columns(3)
                        
                        p_uncal = details.get("raw_uncalibrated_probability") or details.get("provenance", {}).get("raw_uncalibrated_probability")
                        p_cal = details.get("calibrated_probability") or details.get("provenance", {}).get("calibrated_probability")
                        ev_net = details.get("decision_ev_inr") or details.get("provenance", {}).get("decision_ev_inr")

                        if p_uncal is not None:
                            col_p1.metric("Raw P(Recovery)", f"{p_uncal * 100:.1f}%")
                        if p_cal is not None:
                            delta_p = (p_cal - p_uncal) if p_uncal is not None else 0.0
                            col_p2.metric("Calibrated P(Recovery)", f"{p_cal * 100:.1f}%", f"{delta_p * 100:+.1f}% shift" if p_uncal else None)
                        if ev_net is not None:
                            col_p3.metric("Expected Net ROI", f"₹{ev_net:,.2f}")

                        guardrails_passed = details.get("guardrails_passed") or details.get("provenance", {}).get("guardrails_passed", [])
                        if guardrails_passed:
                            st.markdown("**🛡️ Guardrail Invariants Verified:**")
                            for g in guardrails_passed:
                                st.write(f"- ✅ `{g}`")

                        if a.idempotency_key:
                            st.markdown(f"**🔑 Cryptographic SHA-256 Idempotency Key:** `{a.idempotency_key}`")
        else:
            st.info(f"No state transition events logged yet for {selected_txn_id}.")

