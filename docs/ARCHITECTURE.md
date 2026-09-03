# RecoverAI — Technical Architecture Specification
**Submission for Razorpay AI Builder Internship Buildathon (Track 03: AI Revenue Recovery)**

---

## 1. High-Level System Architecture

```mermaid
flowchart TD
    subgraph Client & Merchant Tier
        UI[Streamlit Executive & Workbench UI]
        API_CLIENT[Merchant Webhook / REST Client]
    end

    subgraph API & Routing Layer
        FASTAPI[FastAPI REST Backend]
        CORS[CORS & Auth Middleware]
        FASTAPI --- CORS
    end

    subgraph Data & Contract Ingestion
        GEN[Synthetic Transaction Generator]
        VAL[Data Contract Validator]
        FEAT[Leak-Free Feature Pipeline]
        GEN --> VAL --> FEAT
    end

    subgraph Intelligence & Decision Core
        ML_MODEL[Calibrated Recovery Classifier\nLightGBM + Platt Calibration]
        DEC_ENG[Expected Value Decision Engine\nE[Net ROI] Maximizer]
        HARD_RULES[Deterministic Safety Rules Engine]
        LLM_LAYER[LLM Reasoning & Copywriting Layer\nStructured Pydantic + Fallback]
        
        FEAT --> ML_MODEL
        ML_MODEL --> DEC_ENG
        HARD_RULES --> DEC_ENG
        DEC_ENG --> LLM_LAYER
    end

    subgraph Bounded Agent State Machine
        STATE_MACH[Agent Lifecycle Engine\nDetect -> Diagnose -> Decide -> Act -> Verify]
        GUARDRAILS[Circuit Breaker & Guardrail System\nMax Retries | 24h Cooldown | High Value]
        LLM_LAYER --> STATE_MACH
        STATE_MACH <--> GUARDRAILS
    end

    subgraph Execution & Simulation
        SIM_ENG[Stochastic Payment Rails Simulator\nBank Downtime & Customer Action Model]
        STATE_MACH --> SIM_ENG
    end

    subgraph Storage & Observability
        SQLITE[(SQLite Persistent DB\nWAL Mode)]
        AUDIT[Immutable Event Audit Logger]
        SIM_ENG --> AUDIT
        AUDIT --> SQLITE
    end

    UI <--> FASTAPI
    API_CLIENT <--> FASTAPI
    FASTAPI <--> STATE_MACH
    FASTAPI <--> SQLITE
```

---

## 2. Core Subsystems

### 2.1 Data Ingestion & Contract Validator (`recoverai/data/`)
* **Contract**: Incoming payment failure payloads must conform to the strict `PaymentTransaction` schema.
* **Validation**: `DataContractValidator` enforces positive amounts, valid Indian banking error codes, valid payment rails, and chronological ordering.

### 2.2 Feature Engineering Pipeline (`recoverai/features/`)
* **Features Extracted**:
  - `log_amount`, `log_ltv`, `amount_to_ltv_ratio`: Non-linear customer & transaction scales.
  - `is_maintenance_window`: Midnight bank switch downtime flag (23:00–03:00).
  - `is_salary_window`: High-liquidity days (28–5).
  - `hour_sin`, `hour_cos`, `day_sin`, `day_cos`: Cyclical temporal components.
  - `bank_reliability`: Prior uptime rating of issuer bank.
  - `is_transient_failure`, `is_user_action_required`: High-order failure interaction indicators.
* **Zero Leakage**: Feature transformations and `StandardScaler` are fitted strictly on the training set and persisted in `artifacts/models/scaler.joblib`.

### 2.3 ML Recovery Probability Estimator (`recoverai/models/`)
* **Algorithm**: LightGBM Classifier (with HistGradientBoosting fallback) wrapped in `CalibratedClassifierCV(method='sigmoid', cv=5)`.
* **Calibration & Governance**: Ensures that output probabilities $P(\text{Recovery})$ accurately reflect empirical recovery rates ($ECE = 0.0101$, a 96.3% calibration error reduction over uncalibrated baselines) to govern edge-case expected values and explainability.

### 2.4 Expected Value Decision Engine (`recoverai/decision/`)
* **Optimization Function**:
  $$\mathbb{E}[\text{Net ROI}(\text{Action})] = P(\text{Recovery} \mid \text{Action}) \times \text{Amount} - \text{Cost}(\text{Action}) - \text{CustomerFriction}(\text{Action})$$
* **Candidate Actions**:
  - `SMART_RETRY_DELAYED` (Cost: ₹0.50, Friction: ₹0.00)
  - `SMART_RETRY_IMMEDIATE` (Cost: ₹0.50, Friction: ₹0.00)
  - `WHATSAPP_PAY_LINK` (Cost: ₹0.80, Friction: ₹3.00)
  - `SMS_PAY_LINK` (Cost: ₹0.25, Friction: ₹3.00)
  - `UPI_INTENT_SWITCH` (Cost: ₹0.40, Friction: ₹1.50)
  - `MANUAL_ESCALATION` (Cost: ₹50.00, Friction: ₹0.00)
  - `NO_ACTION` (Cost: ₹0.00, Friction: ₹0.00)

### 2.5 Constrained LLM Reasoning Layer (`recoverai/llm/`)
* **Role**: Structured diagnosis synthesis and personalized, empathetic customer messaging.
* **Safety & Resilience**:
  - Pydantic schema validation (`LLMExplanation`).
  - Graceful deterministic offline fallback engine (`DeterministicFallbackReasoningEngine`) ensuring 100% reliability even when API keys are absent or rate-limited.

### 2.6 Bounded Agent State Machine & Guardrails (`recoverai/agent/`)
* **State Transition Graph**:
  `DETECTED` $\to$ `DIAGNOSED` $\to$ `DECIDED` $\to$ `ACTION_SCHEDULED` / `ACTION_EXECUTED` $\to$ `VERIFIED` $\to$ `RECOVERED` / `FAILED_TERMINAL` / `ESCALATED`.
* **Hard Circuit Breakers**:
  1. *Max Retry Guard*: Exactly 3 recovery attempts permitted per transaction.
  2. *Fatigue Guard*: Maximum 1 communication per customer per 24 hours.
  3. *High-Value Guard*: Transactions $\ge \text{₹}50,000$ automatically routed to human operations.
  4. *Idempotency Guard*: SHA-256 hash tracking prevents double charging or duplicate link dispatch.

---

## 3. Database Schema (`recoverai/storage/`)

### 3.1 `transactions` Table
- `transaction_id` (PK, VARCHAR)
- `merchant_id`, `customer_id`, `amount_inr`, `currency`
- `payment_method`, `bank_code`, `card_network`
- `failure_category`, `gateway_error_code`, `gateway_error_description`
- `retry_attempt_count`, `current_agent_state`, `is_recovered`, `amount_recovered_inr`
- `created_at`, `updated_at`

### 3.2 `recovery_decisions` Table
- `id` (PK, INTEGER)
- `transaction_id` (FK)
- `recommended_action`, `recommended_delay_hours`
- `estimated_recovery_probability`, `expected_recovery_value_inr`
- `intervention_cost_inr`, `net_expected_roi_inr`
- `root_cause_diagnosis`, `merchant_recommendation`, `customer_recovery_message`

### 3.3 `audit_events` Table (Immutable Append-Only Log)
- `id` (PK, INTEGER)
- `transaction_id` (FK)
- `timestamp`, `previous_state`, `current_state`
- `action_type`, `actor`, `idempotency_key`
- `cost_incurred_inr`, `amount_recovered_inr`, `details_json`
