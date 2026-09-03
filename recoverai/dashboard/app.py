"""
RecoverAI: Streamlit Interactive Dashboard & Agent Demonstration.
Submission for Razorpay AI Builder Internship Buildathon (Track 03 - AI Revenue Recovery).
"""

import streamlit as st
from recoverai.storage.database import init_db
from recoverai.config.settings import settings

# Initialize database schema on startup
init_db()

st.set_page_config(
    page_title="RecoverAI — AI Revenue Recovery Agent",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Razorpay styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0c2340;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.2rem;
    }
    .badge-synthetic {
        background-color: #fef3c7;
        color: #92400e;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #fde68a;
    }
    .card {
        padding: 1.2rem;
        background-color: #ffffff;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar Navigation
with st.sidebar:
    st.image("https://razorpay.com/assets/razorpay-logo.svg", width=180)
    st.markdown("### **RecoverAI Agent**")
    st.caption("AI Revenue Recovery Agent — Track 03")
    
    st.markdown("---")
    view_selection = st.radio(
        "Navigation",
        [
            "📊 Executive ROI Overview",
            "⚡ Live Agent Workbench",
            "📜 Case Drilldown & Audit Trail",
            "🧠 ML Explainability & Calibration",
        ],
    )
    st.markdown("---")

    st.markdown("#### **System Diagnostics**")
    model_exists = (settings.MODELS_DIR / "recovery_model.joblib").exists()
    st.write(f"**ML Classifier:** {'🟢 Loaded' if model_exists else '🔴 Missing'}")
    st.write(f"**Reasoning Engine:** `Deterministic + Live LLM`")
    st.write(f"**Safety Guardrails:** `Active (Max 3, 24h Cooldown, ₹50k Escalation)`")
    st.write(f"**Database:** `SQLite (WAL mode)`")

    st.markdown("---")
    st.markdown(
        """
        <div style='font-size: 0.8rem; color: #64748b;'>
        <b>Razorpay AI Builder Buildathon</b><br>
        Track 03: AI Revenue Recovery<br>
        v1.0.0 • Production Benchmark
        </div>
        """,
        unsafe_allow_html=True,
    )

# Top Banner Disclaimer
st.markdown(
    """
    <div style='background-color: #f8fafc; border-left: 4px solid #3395ff; padding: 10px 16px; border-radius: 4px; margin-bottom: 20px;'>
    <span class='badge-synthetic'>⚠️ SYNTHETIC BENCHMARK SIMULATOR</span>
    <span style='font-size: 0.9rem; color: #334155; margin-left: 8px;'>
    All transactions, failure codes, and customer responses are generated under empirical statistical distributions for benchmarking. No real customer credentials or proprietary Razorpay production data are used.
    </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# View Dispatcher
from recoverai.dashboard.views.executive import render_executive_view
from recoverai.dashboard.views.workbench import render_workbench_view
from recoverai.dashboard.views.audit_view import render_audit_view
from recoverai.dashboard.views.ml_explainability import render_ml_explainability_view

if view_selection == "📊 Executive ROI Overview":
    render_executive_view()
elif view_selection == "⚡ Live Agent Workbench":
    render_workbench_view()
elif view_selection == "📜 Case Drilldown & Audit Trail":
    render_audit_view()
elif view_selection == "🧠 ML Explainability & Calibration":
    render_ml_explainability_view()
