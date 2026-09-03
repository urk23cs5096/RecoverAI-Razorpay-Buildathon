# 30-Seed Statistical Robustness & Attribution Summary
> **Generated Programmatically from `robustness_30seed_results.csv`**

---

## 1. Overview & Evaluation Parameters
- **Dataset Evaluated**: Held-Out Test Set (`data/processed/test.csv`)
- **Total Transactions per Run**: 1,500
- **Total Gross Revenue at Risk per Run**: ₹36,351,022.68
- **Number of Synchronized Random Seeds**: 30
- **Seeds Tested**: `42, 143, 244, 345, 446, 547, 648, 749, 850, 951, 1052, 1153, 1254, 1355, 1456, 1557, 1658, 1759, 1860, 1961, 2062, 2163, 2264, 2365, 2466, 2567, 2668, 2769, 2870, 2971`

---

## 2. Macro Action-Design Lift (Multi-Rail Action Design vs. Rule-Based Dunning Baseline)
*Evaluates the value of multi-rail actions (WhatsApp 1-Click links, mandate re-auth links, dynamic switch backoffs, and fatigue cooldowns) against standard gateway dunning rules.*

| Metric | Empirical Value |
| :--- | :--- |
| **Number of Seeds** | **30** |
| **Mean Macro Net Lift** | **₹+23,674,637.36** |
| **Standard Deviation (Std Dev)** | **₹758,456.64** |
| **Standard Error of Mean (SEM)** | **₹138,474.60** |
| **95% Confidence Interval** | **[₹+23,391,425.00, ₹+23,957,849.72]** |
| **Min / Max Lift Observed** | ₹+22,410,307.10 / ₹+25,781,070.81 |
| **Positive Lift Runs** | 30 / 30 (100.0%) |
| **Paired $t$-Test Statistic** | $t = 170.9674$ |
| **Paired $t$-Test $p$-Value** | $p = 4.0855e-45$ |
| **Statistical Significance ($\\alpha = 0.05$)** | **YES (Overwhelmingly Significant)** |

---

## 3. Precision ML Tiering Lift (RecoverAI Full Agent vs. RecoverAI_Actions_No_ML Ablation)
*Evaluates the incremental value of calibrated LightGBM propensity scoring and Expected Value tiering when both arms share the exact same multi-rail action space and identical 24h fatigue guardrails.*

| Metric | Empirical Value |
| :--- | :--- |
| **Number of Seeds** | **30** |
| **Single-Run Snapshot (`seed=42`)** | +₹7,09,501.32 (+₹7.10 Lakhs, +2.0% rate) |
| **30-Seed Mean Incremental Lift** | **₹-157,855.93** |
| **Standard Deviation (Std Dev)** | **₹637,313.96** |
| **Standard Error of Mean (SEM)** | **₹116,357.08** |
| **95% Confidence Interval** | **[₹-395,832.88, ₹+80,121.01]** *(Crosses zero)* |
| **Min / Max Lift Observed** | ₹-1,167,277.93 / ₹+1,294,902.76 |
| **Positive Lift Runs** | 12 / 30 (40.0%) |
| **Paired $t$-Test Statistic** | $t = -1.3567$ |
| **Paired $t$-Test $p$-Value** | $p = 0.1854$ |
| **Wilcoxon Signed-Rank Test** | $W = 163.0, p = 0.1579$ |
| **Statistical Significance ($\\alpha = 0.05$)** | **NO (Not Statistically Significant)** |

---

## 4. Key Engineering Takeaways
1. **Multi-Rail Architecture Drives Business Lift**: The primary recovery leap (+₹2.37 Cr net GMV, $p = 4.09 \times 10^{-45}$) is statistically robust and driven by multi-rail routing (WhatsApp, UPI Intent, dynamic backoffs).
2. **ML Calibration & Governance Role**: Incremental ML tiering variation falls within stochastic simulation noise ($p = 0.1854$, CI crosses zero). The ML layer's primary production value is **probability calibration (ECE: 0.0101)**, automated feature attribution, and edge-case margin safety.
