# RecoverAI — 5-Minute Timed Pitch Script
**Razorpay AI Builder Internship Buildathon (Track 03: AI Revenue Recovery)**

---

### [0:00 – 0:30] 1. The Problem: Silent Revenue Bleed in Indian Payments
> *"In India’s fast-growing digital economy, recurring SaaS platforms and D2C brands quietly lose between 3% to 7% of their gross merchandise value every single month. But here's the kicker: over 70% of these failed payments aren't caused by bankrupt customers or fraud. They fail due to preventable friction: midnight bank switch outages (`U19`), expired recurring e-mandates (`U30`), 3DS authentication timeouts, and daily card limits. For a merchant doing ₹2 Crore monthly GMV, that is ₹10 to ₹14 Lakhs of pure revenue slipping away every month."*

---

### [0:30 – 1:15] 2. Why Existing Solutions Fall Short
> *"Today, how do payment gateways and merchants handle payment failures? They rely on two flawed extremes:*
> *1. **Blind Auto-Retries**: Payment gateways retry every failure immediately. When a bank's switch is down at midnight, an immediate retry fails 92% of the time, incurs gateway retry surcharges, and burns card network trust scores.*
> *2. **Aggressive Dumb Dunning**: Sending generic, annoying payment reminders to high-value customers when the fault was just a temporary bank glitch.*
> 
> *What merchants desperately need is an intelligent, cost-aware agent that understands **why** a payment failed, calculates the **Expected Value** of intervening, and acts within strict **safety guardrails**. That is why we built **RecoverAI**."*

---

### [1:15 – 2:45] 3. Live Product & Workbench Demo *(Screen Share UI & CLI Walkthrough)*
> *"Let's see RecoverAI execute live.*
> 
> *(Running `python run_demo.py` on real held-out incident `txn_102116`)*
> *Watch the agent execute its 8-stage pipeline in under 5 seconds on a real ₹1,19,334.92 B2B enterprise payment failure:*
> 1. **Incident Ingestion & Features**: The agent ingests the `INSUFFICIENT_FUNDS` error (`U66`) and transforms 14 leak-free features in under 10ms.
> 2. **Probability Calibration in Action**: The raw tree model underconfidently predicted 59.4%. Platt scaling corrects this to **68.04%** ($ECE = 0.0101$).
> 3. **Expected Value Financial Decision**: Uncalibrated probabilities would have selected a delayed WhatsApp link (Net EV: ₹79,424). Calibrated EV selects **`MANUAL_ESCALATION`** (Net EV: **₹1,04,964**), capturing an extra **+₹25,539.85 in expected ROI**.
> 4. **Deterministic Safety Guardrails**: Circuit breakers instantly verify Max Retries ($0 < 3$), 24h fatigue cooldown, and the ₹50,000+ high-value threshold to approve concierge escalation.
> 5. **Execution & Physical Rails Verification**: The agent drafts empathetic recovery copy, issues a unique SHA-256 idempotency key (`3ed490b1...`), and executes via the physical simulator—recovering the full **₹1,19,334.92 gross revenue** with ₹50 direct ops cost and ₹0 friction (Net: **+₹1,19,284.92**).
> 6. **Immutable Provenance**: All math, calibrated probabilities, and guardrails are cryptographically logged to the SQLite audit trail.
> 
> *In our interactive Workbench, this same engine handles bank downtime with dynamic maintenance backoffs and 3DS drops with 1-click WhatsApp pay links."*

---

### [2:45 – 3:30] 4. Explainable AI/ML Architecture
> *"Let’s look under the hood. RecoverAI is built on three core pillars:*
> 1. **Calibrated Machine Learning**: Standard boosted trees produce uncalibrated probabilities that distort financial calculations. We trained a LightGBM classifier with **Platt Scaling (`CalibratedClassifierCV`)**, driving Expected Calibration Error down from 0.2710 to **0.0101** (a 96.3% calibration improvement).
> 2. **Mathematical Expected Value Engine**: We maximize $\mathbb{E}[\text{Net Value}] = P(\text{Recovery}) \times \text{Amount} - \text{Intervention Cost} - \text{Customer Friction Penalty}$. Calibrated probabilities ensure cost-aware edge-case suppression to protect merchant margins.
> 3. **Deterministic Safety State Machine**: The LLM has zero authority to execute financial actions directly. It is strictly bounded by hard invariants: a max limit of 3 retries, a 24-hour customer communication cooldown, and mandatory human escalation for high-value transactions over ₹50,000."*

---

### [3:30 – 4:15] 5. Measured Evidence & Honest Value Attribution *(Screen Share Dashboard)*
> *"We evaluated RecoverAI on a benchmark of 10,000 transactions across 5 merchant verticals and ran a rigorous **30-seed counterfactual robustness evaluation** on 1,500 held-out test transactions (₹3.64 Crore at risk).*
> 
> *Here is the honest breakdown of where the value comes from:*
> - **Macro Multi-Rail Action Design (+₹2.37 Crore Net Lift, $p < 0.001$)**: The overwhelming driver of revenue recovery is the multi-rail action architecture—providing 1-click WhatsApp links for 3DS drops, mandate re-auth links, and dynamic switch backoffs. This delivers a statistically robust **+₹2.37 Crore Net Revenue Lift** (95% CI: [+₹2.34 Cr, +₹2.40 Cr], $p = 4.09 \times 10^{-45}$) over standard rule-based dunning (17.1% revenue recovery rate baseline, 17.9% transaction count).
> - **The Role of ML Calibration**: In our multi-seed ablation check, fine-grained ML tiering variance falls within simulation noise across seeds (mean -₹1.58L, $p=0.185$). The real production value of the ML layer is **probability calibration (ECE: 0.0101)**, automated feature attribution, and edge-case margin protection.
> - **Safety & Cost Efficiency**: By replacing blind retries and redundant SMS blasts with fatigue cooldowns and smart routing, RecoverAI reduces operational recovery waste while safeguarding customer relationships."*

---

### [4:15 – 5:00] 6. How It Scales & Conclusion
> *"RecoverAI is built production-ready with a high-performance **FastAPI backend**, comprehensive **Pydantic schema contracts**, and a **100% test-covered suite**.*
> 
> *For Razorpay, this architecture can plug directly into webhooks (`payment.failed`, `subscription.charged`), orchestrating intelligent smart-routing and 1-click WhatsApp recovery across lakhs of merchants.*
> 
> *Revenue recovery isn't about spamming users or blindly hammering bank APIs—it's about mathematical expected value, intelligent timing, and safe agentic execution. That is RecoverAI. Thank you!"*
