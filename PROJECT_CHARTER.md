# PROJECT CHARTER: RecoverAI
**AI Revenue Recovery Agent for Subscription & Digital Merchants**  
*Submission for Razorpay AI Builder Internship Buildathon — Track 03 (AI Revenue Recovery)*

---

## 1. Executive Summary
**RecoverAI** is an intelligent, cost-aware, and safety-bounded autonomous revenue recovery agent designed to recover lost GMV from failed digital payments, dropped checkout attempts, and expired recurring mandates. 

Instead of relying on naive, blind auto-retries that irritate customers, trigger bank rate limits, and incur gateway fees, RecoverAI pairs a **calibrated machine learning propensity model** with an **Expected Value Decision Engine** ($\mathbb{E}[\text{Net Recovery Value}]$) and a **bounded, deterministic safety state machine**. An integrated, guardrailed LLM reasoning layer synthesizes failure diagnostics and drafts empathetic, contextualized payment recovery communications.

---

## 2. Problem Statement & Market Context
In India's digital payments ecosystem (UPI, Cards, NetBanking, e-NACH/Auto-Debit mandates), merchants lose **3% to 7% of gross transaction value** to preventable payment failures:
1. **Transient Bank / Issuer Switch Outages (`U19`, `91`, `3DS_TIMEOUT`)**: Failures where the customer has sufficient funds, but the transaction fails due to peak gateway load or maintenance windows.
2. **Involuntary Subscription Mandate Churn (`U30`, `MANDATE_INACTIVE`)**: Recurring subscription charges that fail due to expired mandates, card renewals, or temporary balance shortfalls.
3. **Friction-Heavy Checkout Drop-offs**: Customers who experience payment failures and abandon their cart without retrying unless offered an alternative payment rail (e.g., switching from a failing card gateway to 1-click UPI Intent).
4. **Suboptimal Naive Retries**: Typical merchant gateways trigger blind immediate retries (e.g., 3 retries in 24 hours). This causes repeated bank rejections, higher customer churn, and payment network penalty surcharges.

---

## 3. Target Persona
* **Primary Persona**: Engineering and Finance Operations Leads at mid-market SaaS platforms and high-volume D2C subscription brands processing **₹50 Lakh to ₹5 Crore monthly GMV** via Razorpay.
* **Core Pain Point**: Lost revenue from involuntary payment failures, lack of actionable root-cause visibility, and fear of damaging customer relationships with aggressive or poorly timed payment reminders.

---

## 4. Key Architectural Differentiators

| Capability | Standard Gateway / Naive Dunning | RecoverAI Agentic Approach |
| :--- | :--- | :--- |
| **Retry Strategy** | Blind immediate or static daily retries | **Dynamic Timing Optimization** based on issuer uptime patterns and customer activity history |
| **Decision Logic** | Static IF/ELSE rules or uncalibrated ML | **Expected Value ($\mathbb{E}[\text{Value}]$) Maximizer** balancing recovery odds, fees, and churn risk |
| **Action Channels** | Single channel (Card auto-retry only) | **Multi-Rail Orchestration**: Smart Retry, 1-Click WhatsApp Pay Link, UPI Intent Switch, Merchant Escalation |
| **LLM Integration** | None or unconstrained hallucination-prone prompt | **Constrained Reasoning Layer**: Structured Pydantic output with deterministic fallback |
| **Safety Guardrails** | Minimal rate-limiting | **Hard Invariants**: Max 3 attempts, 24h messaging cooldown, idempotency, ₹50k+ human escalation |

---

## 5. Scope & Boundary Contract

### What is Simulated (Synthetic Benchmark)
* **Transaction & Failure Dataset**: 10,000 synthetic transaction records statistically modeling Indian payment failure distributions, issuer downtime schedules, customer payment preferences, and historical LTV.
* **Bank & Customer Response Simulator**: Stochastic response engine modeling recovery outcomes, payment link click-throughs, and payment rail switch completions based on empirical industry benchmarks.
* **Disclaimer**: Clearly tagged with `[SYNTHETIC BENCHMARK]` across all UI views, logs, and evaluation reports.

### What is Production-Grade & Real
* **Backend API**: Production-ready FastAPI REST service with schema validation, healthchecks, and OpenAPI documentation.
* **ML & Feature Pipeline**: Fully reproducible feature engineering, train/validation/test split with zero target leakage, calibrated LightGBM classifier, and comprehensive model evaluation.
* **Bounded State Machine**: Thread-safe, auditable state machine with strict safety guardrails, idempotency validation, and circuit breakers.
* **Interactive UI**: Multi-view Streamlit dashboard with executive ROI summaries, real-time agent workbench, and per-transaction audit logs.
* **Test Suite**: Automated unit, integration, and safety invariant tests.

---

## 6. Business Impact & Evaluation Metrics

1. **Net Recovered Revenue ($\Delta \text{₹}$)**: Total gross recovered revenue minus recovery operational costs (gateway fees, SMS/WhatsApp API costs).
2. **Recovery Rate Lift (%)**: Percentage point increase in successfully recovered transactions over the *Naive Immediate 3x Retry* baseline.
3. **Recovery Precision**: Percentage of triggered interventions that resulted in successful payment recovery (minimizing unnecessary friction).
4. **Customer Friction Reduction**: Reduction in unnecessary customer communications and redundant bank API calls.
5. **Model Calibration Score (Brier Score & ECE)**: Ensuring predicted recovery probabilities accurately reflect real recovery frequencies.

---

## 7. Hard Safety Invariants
1. **Never Infinitely Retry**: Hard limit of **3 recovery attempts** per failed transaction.
2. **Customer Communication Cooldown**: Maximum of **1 notification per 24-hour window** per customer across all channels.
3. **Idempotency Guarantee**: Every intervention carries a unique idempotency key to prevent duplicate payment link generation or charge attempts.
4. **Circuit Breaker for High-Value Transactions**: Any transaction $\ge \text{₹}50,000$ requires explicit merchant human confirmation before automated recovery actions are dispatched.
5. **LLM Fallback Invariant**: In case of LLM API unavailability, rate limit, or schema failure, the system falls back seamlessly to deterministic rule-based explanations with 0% downtime.
