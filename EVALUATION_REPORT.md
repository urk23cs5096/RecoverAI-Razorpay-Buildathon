# RecoverAI — Machine Learning & Business Impact Evaluation Report
**Track 03: AI Revenue Recovery (Razorpay AI Builder Buildathon)**

---

## 1. Executive Summary
Payment failure recovery is fundamentally an **Expected Value optimization problem under cost, rail availability, and customer fatigue constraints**, not a simple classification problem.

Standard payment gateways either execute blind auto-retries (which burn bank rate limits, trigger gateway surcharges, and annoy customers) or apply rigid static rules (which fail on 3DS drops, mandate updates, and maintenance windows). **RecoverAI** introduces a **calibrated Gradient Boosted Decision Tree (LightGBM + Platt Scaling)** coupled with a **Tiered Expected Value Decision Engine ($\mathbb{E}[\text{Net Value}]$)** and deterministic safety guardrails.

This report provides an empirical evaluation across 10,000 synthetic payment failure transactions comparing RecoverAI against industry-standard baselines and an explicit **No-ML Tiered Ablation Arm** on held-out test data with **zero data leakage** and a **100% reconciled financial ledger**.

> **⚠️ SYNTHETIC BENCHMARK DISCLAIMER**: All datasets, transaction logs, bank switch outage profiles, and customer response distributions are synthetically generated for benchmark evaluation based on authentic Indian payment industry parameters.

---

## 2. Dataset Profile & Split Methodology

* **Total Transaction Volume**: 10,000 payment failure records spanning SaaS B2B, D2C eCommerce, EdTech Subscriptions, OTT Media, and B2B Invoicing.
* **Payment Rails**: UPI, Credit/Debit Cards (Visa, Mastercard, RuPay, Amex), NetBanking, e-Mandate Cards, e-Mandate UPI.
* **Bank Routing Coverage**: HDFC (28%), ICICI (24%), SBI (20%), AXIS (12%), Kotak (8%), Others (8%).
* **Split Strategy**: Chronological 70% Train (7,000 rows), 15% Validation (1,500 rows), 15% Test (1,500 rows).
* **Zero Leakage Guarantee**: Preprocessing scalers and encoders are fitted strictly on the training set and serialized (`scaler.joblib`) for inference.

---

## 3. Empirical Model Comparison (Held-Out Test Set: ML Classification & Probability Calibration)

Evaluated on 1,500 held-out test transactions representing **₹3,63,51,022.68** in gross revenue at risk:

| Model Architecture | ROC-AUC | PR-AUC | Precision | Recall | F1 Score | Brier Score (↓) | ECE (↓) | Gross Recovered (₹) | Net Recovered (₹) | Revenue Recovery Rate (%) | Interventions |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule-Based Heuristic** | 0.5693 | 0.8129 | **0.8806** | 0.3374 | 0.4878 | 0.4389 | 0.5011 | ₹92,20,555.91 | ₹92,20,041.71 | 25.4% | 444 |
| **Logistic Regression Baseline** | **0.6173** | 0.8435 | 0.8531 | 0.4409 | 0.5813 | 0.2486 | 0.2710 | ₹1,25,11,545.09 | ₹1,25,10,801.89 | 34.4% | 599 |
| **RecoverAI Calibrated Classifier** | 0.6147 | **0.8450** | 0.7727 | **1.0000** | **0.8718** | **0.1703** | **0.0101** | **₹2,79,99,591.18** | **₹2,79,97,368.18** | **77.0%** | **1,500** |

---

## 4. Deep Dive: Probability Calibration & Classification Thresholds

1. **Probability Distribution**: The calibrated classifier produces non-degenerate, continuous probabilities ranging from **0.5509 to 0.9471** (mean: **0.7626**, std: **0.0785**), reflecting genuine distinctions across ticket sizes, issuer switch uptime, and customer loyalty tiers.
2. **High Base Recovery Under Optimal Routing**: Because optimal interventions (e.g. delayed retry after switch maintenance, 1-click WhatsApp links) have high intrinsic recovery rates, all predicted probabilities naturally exceed 0.50, yielding Recall = 1.0000 at the standard arbitrary 0.50 cutoff.
3. **Threshold Sensitivity**:
   * At threshold `0.70` (near base rate): Precision = **0.7959**, Recall = **0.7972**, F1 = **0.7966**.
   * At threshold `0.80`: Precision = **0.8756**, Recall = **0.3279**, F1 = **0.4771**.

---

## 5. Physical Simulation & Fully-Reconciled Financial Ledger (1,500 Test Cases, ₹3.64 Cr at Risk)

The financial ledger reconciles every paisa: $\text{Net Revenue Recovered} = \text{Gross Recovered} - (\text{Direct Operational Costs} + \text{Customer Friction Costs})$.

| Policy Strategy | Recovered Txns (Count & %) | Revenue Recovery Rate (%) | Gross Recovered (₹) | Direct Operational Costs (₹) | Customer Friction Costs (₹) | Total Deductions (₹) | Net Revenue Recovered (₹) | Interventions Triggered |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **RecoverAI (Full Agent with ML Tiering)** | **1,147 / 1,500 (76.5%)** | **82.3%** | **₹2,99,02,875.73** | **₹23,892.90** | **₹1,233.00** | **₹25,125.90** | **₹2,98,77,749.83** | **1,480** |
| **RecoverAI_Actions_No_ML (Ablation: Uniform Prior $P=0.65$)** | 1,123 / 1,500 (74.9%) | 80.3% | ₹2,91,97,092.81 | ₹26,897.30 | ₹1,947.00 | ₹28,844.30 | ₹2,91,68,248.51 | 1,437 |
| **Rule-Based Heuristic (Standard Dunning)** | 269 / 1,500 (17.9%) | 17.1% | ₹62,32,789.22 | ₹480.75 | ₹1,197.00 | ₹1,677.75 | ₹62,31,111.47 | 1,161 |
| **Naive Immediate 3x Retry** | 85 / 1,500 (5.7%) | 5.4% | ₹19,67,263.09 | ₹2,250.00 | ₹0.00 | ₹2,250.00 | ₹19,65,013.09 | 4,500 |
| **No Intervention (Baseline)** | 0 / 1,500 (0.0%) | 0.0% | ₹0.00 | ₹0.00 | ₹0.00 | ₹0.00 | ₹0.00 | 0 |

---

## 6. Honest Decomposition of Value Attribution & Multi-Seed Statistical Robustness

To rigorously evaluate where the recovery value originates, we performed an ablation study across **30 distinct random seeds with synchronized paired seeds** comparing:
1. **Full RecoverAI Agent** (Calibrated LightGBM + Expected Value Tiering + Safety Guardrails)
2. **No-ML Multi-Rail Ablation Arm (`RecoverAI_Actions_No_ML`)** (Identical Multi-Rail Action Candidate Set & Guardrails, Uniform Prior $P=0.65$)
3. **Standard Rule-Based Dunning Baseline** (Immediate retry for downtime, generic SMS for balance, drop 3DS/mandates)

```
========================================================================================
         30-SEED EMPIRICAL STATISTICAL ROBUSTNESS & ATTRIBUTION DECOMPOSITION
========================================================================================
1. MACRO ACTION-DESIGN LIFT (Multi-Rail Action Design vs. Rule-Based Dunning):
   • Standard Dunning Rules Net Recovery:                ₹62,31,111.47  (17.1% revenue rate)
   • No-ML Tiered Multi-Rail Baseline Net Recovery:      ₹2,91,68,248.51 (80.3% revenue rate)
   -------------------------------------------------------------------------------------
   • Mean Macro Net Revenue Lift (30 Seeds):            +₹2,36,74,637.36 (+₹2.37 Cr Net Lift)
   • 95% Confidence Interval:                           [+₹2,33,91,425.00, +₹2,39,57,849.72]
   • Paired t-test:                                     t = 170.97, p = 4.09e-45 (p < 0.001)
   • Statistical Verdict:                               OVERWHELMINGLY STATISTICALLY SIGNIFICANT

2. PRECISION ML TIERING LIFT (Full ML Tiering vs. No-ML Multi-Rail Ablation Arm):
   • Single-Seed Snapshot (Seed=42):                    +₹7,09,501.32 (+₹7.10 Lakhs, +2.0% revenue rate)
   • 30-Seed Empirical Distribution:
     - Mean Incremental ML Net Lift:                    -₹1,57,855.93 (-₹1.58 Lakhs)
     - Standard Deviation (Std Dev):                    ₹6,37,313.96
     - Standard Error of the Mean (SEM):                ₹1,16,357.08
     - 95% Confidence Interval:                         [-₹3,95,832.88, +₹80,121.01]
     - Min / Max Lift Observed:                         -₹11.67 Lakhs / +₹12.95 Lakhs
     - Positive Lift Realizations:                      12 / 30 seeds (40.0%)
     - Paired t-test:                                   t = -1.3567, p = 0.1854 (p > 0.05)
     - Wilcoxon Signed-Rank Test:                       W = 163.0, p = 0.1580 (p > 0.05)
   • Statistical Verdict:                               NOT STATISTICALLY SIGNIFICANT (CI crosses zero)
========================================================================================
```

### Honest Engineering Interpretation of Value Attribution:
1. **The Primary Value Engine is Multi-Rail Action Design (+₹2.37 Cr)**:
   The massive leap from ~17.1% to ~80.3% revenue recovery rate (+₹2.37 Cr net recovered GMV, $p = 4.09 \times 10^{-45}$) is causally driven by the **agentic multi-rail action architecture** (1-Click WhatsApp pay links for 3DS drops, mandate re-auth links, dynamic switch maintenance backoffs, and the 24-hour customer communication fatigue cooldown).
2. **Fine-Grained ML Tiering operates near the Action-Space Frontier**:
   While the single seed run (seed=42) produced +₹7.10 Lakhs due to routing 43 repeat dropouts to Tier 2 in-app UPI intent switches under fatigue cooldowns, multi-seed evaluation across 30 seeds demonstrates that the 95% CI $[-₹3.96\text{L}, +₹0.80\text{L}]$ crosses zero with $p = 0.1854$. Fine-grained ML probability score differences between Tier 2 and Tier 3 actions produce variances that fall within stochastic simulation noise once optimal action channels and deterministic guardrails are in place.
3. **Takeaway for Production Architecture**:
   In high-throughput fintech infrastructure, investment should prioritize expanding the **action channel surface area** (UPI Intent, WhatsApp 1-Click, e-NACH re-auth, SMS) and **deterministic safety invariants** (fatigue cooldowns, maintenance calendars), while ML models provide structured calibration and automated feature monitoring.

---

## 7. Feature Attribution (Top Predictive Signals)

Normalized Tree Gain from Calibrated LightGBM:
1. `is_transient_failure` (30.06%): Distinguishes momentary switch outages from permanent card limit/mandate expirations.
2. `log_ltv` (13.33%): Customer lifetime value determines margin tolerance for higher-touch communication channels.
3. `log_amount` (10.49%): Transaction ticket size governing Expected Value vs. operational channel cost.
4. `bank_reliability` (9.35%): Issuer switch reliability score (HDFC/ICICI vs. regional PSU switches).
5. `amount_to_ltv_ratio` (6.38%): Customer churn risk profile.
6. `is_bank_downtime` (5.29%): Identifies core switch downtime necessitating delayed backoff.
7. `is_insufficient_funds` (4.41%): Triggers salary cycle alignment (28th–5th of month).
