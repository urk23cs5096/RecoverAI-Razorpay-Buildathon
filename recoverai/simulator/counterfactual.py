"""
Counterfactual Benchmark Runner for Policy Comparison.

Runs side-by-side simulations of:
1. RecoverAI (Full Agent with Calibrated ML Model + Tiered Expected Value Optimization + Guardrails)
2. RecoverAI_Actions_No_ML (Ablation Arm: Same Tiered Candidate Space & Guardrails, but using uniform probability P=0.65)
3. Rule-Based Heuristic (Standard Gateway Dunning Rules)
4. Naive Immediate 3x Retry (Blind Immediate Retries)
5. Zero Intervention (No Action)

Computes gross/net revenue recovered, operational costs, customer friction, and empirical attribution.
"""

import logging
from typing import List, Dict, Any
import pandas as pd
import numpy as np

from recoverai.data.schemas import (
    PaymentTransaction,
    InterventionType,
    FailureCategory,
    RecoveryDecision,
    CustomerTier,
    CommunicationChannel,
)
from recoverai.agent.workflow import RecoveryAgentWorkflow
from recoverai.decision.rules import BusinessRuleEngine
from recoverai.decision.engine import INTERVENTION_SPECS
from recoverai.simulator.engine import RecoverySimulatorEngine
from recoverai.config.settings import settings

logger = logging.getLogger("recoverai.simulator.counterfactual")


class NoMLDecisionEngine:
    """Ablation decision engine using uniform prior probability (P=0.65) without ML features."""

    def decide_intervention(self, txn: PaymentTransaction) -> RecoveryDecision:
        amount = txn.amount_inr
        forced_action, rule_reason, forced_delay = BusinessRuleEngine.apply_hard_rules(txn)
        if forced_action is not None:
            spec = INTERVENTION_SPECS[forced_action]
            cost = spec["cost_inr"] + spec["friction_inr"]
            prob = 0.88 if forced_action == InterventionType.MANUAL_ESCALATION else 0.0
            return RecoveryDecision(
                transaction_id=txn.transaction_id,
                recommended_action=forced_action,
                recommended_delay_hours=forced_delay,
                estimated_recovery_probability=prob,
                expected_recovery_value_inr=prob * amount,
                intervention_cost_inr=cost,
                net_expected_roi_inr=prob * amount - cost,
                action_confidence=prob,
                safety_rule_applied=rule_reason,
                reasoning_summary="Forced by hard safety rule",
            )

        base_prob = 0.65  # Uniform static prior (No ML)
        forbidden = BusinessRuleEngine.get_forbidden_actions(txn)
        best_act = InterventionType.NO_ACTION
        best_roi = -float("inf")
        best_prob = 0.0
        best_cost = 0.0
        best_delay = 0

        # Uniform friction penalty without LTV scaling
        ltv_friction_scale = 1.0

        for act, spec in INTERVENTION_SPECS.items():
            if act == InterventionType.NO_ACTION or act in forbidden:
                continue
            if txn.failure_category not in spec["supported_categories"]:
                continue
            
            p = base_prob * spec["prob_multiplier"]
            cost = spec["cost_inr"]
            fric = spec["friction_inr"] * ltv_friction_scale
            delay = 0

            if txn.failure_category == FailureCategory.BANK_DOWNTIME:
                if act == InterventionType.SMART_RETRY_DELAYED:
                    p = 0.65 * 1.15
                    delay = max(1, 5 - txn.hour_of_day) if txn.hour_of_day in [23, 0, 1, 2, 3] else 1
                elif act == InterventionType.SMART_RETRY_IMMEDIATE:
                    p = 0.07

            elif txn.failure_category == FailureCategory.AUTH_FAILED_3DS:
                if act == InterventionType.WHATSAPP_PAY_LINK:
                    p = 0.65 * 1.12
                elif act == InterventionType.UPI_INTENT_SWITCH:
                    p = 0.65 * 1.08

            elif txn.failure_category == FailureCategory.MANDATE_EXPIRED:
                if act == InterventionType.WHATSAPP_PAY_LINK:
                    p = 0.65 * 1.12

            elif txn.failure_category == FailureCategory.INSUFFICIENT_FUNDS:
                delay = 24 if txn.day_of_month >= 28 else 48
                if act == InterventionType.WHATSAPP_PAY_LINK:
                    p = 0.65 * 1.12

            elif txn.failure_category == FailureCategory.CARD_LIMIT_EXCEEDED:
                # Without ML, No-ML arm always picks default UPI switch
                if act == InterventionType.UPI_INTENT_SWITCH:
                    p = 0.65 * 1.08

            elif txn.failure_category == FailureCategory.NETWORK_ERROR:
                if act == InterventionType.SMART_RETRY_IMMEDIATE:
                    p = 0.65 * 1.10

            net = (p * amount) - cost - fric
            if net > best_roi:
                best_roi = net
                best_act = act
                best_prob = p
                best_cost = cost + fric
                best_delay = delay

        return RecoveryDecision(
            transaction_id=txn.transaction_id,
            recommended_action=best_act,
            recommended_delay_hours=best_delay,
            estimated_recovery_probability=round(best_prob, 4),
            expected_recovery_value_inr=round(best_prob * amount, 2),
            intervention_cost_inr=round(best_cost, 2),
            net_expected_roi_inr=round(best_roi, 2),
            action_confidence=round(best_prob, 4),
            reasoning_summary="Selected by uniform prior",
        )


class CounterfactualBenchmarkRunner:
    """Executes multi-policy simulations on transaction batches."""

    def __init__(
        self,
        agent_workflow: RecoveryAgentWorkflow = None,
        simulator_engine: RecoverySimulatorEngine = None,
        seed: int = 42,
    ):
        self.workflow_ml = agent_workflow or RecoveryAgentWorkflow()
        
        # Ablation workflow with No-ML decision engine (subject to identical safety guardrails)
        self.workflow_noml = RecoveryAgentWorkflow()
        self.workflow_noml.decision_engine = NoMLDecisionEngine()
        
        sim_seed = simulator_engine.seed if simulator_engine is not None else seed
        self.simulator_ml = simulator_engine or RecoverySimulatorEngine(seed=sim_seed)
        self.simulator_noml = RecoverySimulatorEngine(seed=sim_seed)
        self.simulator_rule = RecoverySimulatorEngine(seed=sim_seed)
        self.simulator_naive = RecoverySimulatorEngine(seed=sim_seed)

    def run_benchmark(self, transactions: List[PaymentTransaction]) -> Dict[str, Any]:
        """
        Runs counterfactual comparison across all 5 distinct policies on the given transactions.
        Each policy executes its own distinct action decision under identical guardrails.
        """
        self.workflow_ml.guardrails.reset_state()
        self.workflow_noml.guardrails.reset_state()

        total_at_risk = sum(t.amount_inr for t in transactions)
        total_count = len(transactions)

        # Policy 1: RecoverAI Agent (Full ML + Expected Value)
        recov_results = {"gross_recovered": 0.0, "costs": 0.0, "friction": 0.0, "recovered_count": 0, "interventions": 0}
        
        # Policy 2: RecoverAI_Actions_No_ML (Ablation: Uniform Prior P=0.65, Identical Guardrails)
        noml_results = {"gross_recovered": 0.0, "costs": 0.0, "friction": 0.0, "recovered_count": 0, "interventions": 0}

        # Policy 3: Rule-Based Heuristic (Standard Gateway Dunning Rules)
        rule_results = {"gross_recovered": 0.0, "costs": 0.0, "friction": 0.0, "recovered_count": 0, "interventions": 0}

        # Policy 4: Naive Immediate 3x Auto-Retry
        naive_results = {"gross_recovered": 0.0, "costs": 0.0, "friction": 0.0, "recovered_count": 0, "interventions": 0}

        for txn in transactions:
            # -------------------------------------------------------------
            # Policy 1: RecoverAI Agent (Full ML + Tiered Expected Value Engine + Guardrails)
            # -------------------------------------------------------------
            agent_out = self.workflow_ml.process_incident(txn)
            decision = agent_out["decision"]
            
            if agent_out["action_executed"]:
                sim_out = self.simulator_ml.simulate_outcome(
                    txn=txn,
                    action=decision.recommended_action,
                    delay_hours=decision.recommended_delay_hours,
                )
                if sim_out["is_recovered"]:
                    recov_results["gross_recovered"] += sim_out["amount_recovered_inr"]
                    recov_results["recovered_count"] += 1
                recov_results["costs"] += sim_out["operational_cost_inr"]
                recov_results["friction"] += sim_out["friction_penalty_inr"]
                recov_results["interventions"] += 1

            # -------------------------------------------------------------
            # Policy 2: RecoverAI_Actions_No_ML (Ablation: Policy Rules, Uniform Prior, Same Guardrails)
            # -------------------------------------------------------------
            noml_out = self.workflow_noml.process_incident(txn)
            noml_decision = noml_out["decision"]
            
            if noml_out["action_executed"]:
                sim_noml = self.simulator_noml.simulate_outcome(
                    txn=txn,
                    action=noml_decision.recommended_action,
                    delay_hours=noml_decision.recommended_delay_hours,
                )
                if sim_noml["is_recovered"]:
                    noml_results["gross_recovered"] += sim_noml["amount_recovered_inr"]
                    noml_results["recovered_count"] += 1
                noml_results["costs"] += sim_noml["operational_cost_inr"]
                noml_results["friction"] += sim_noml["friction_penalty_inr"]
                noml_results["interventions"] += 1

            # -------------------------------------------------------------
            # Policy 3: Rule-Based Heuristic (Standard Gateway Behavior)
            # - Bank downtime: Immediate retry (delay=0, fails during maintenance)
            # - Insufficient funds: Generic SMS (delay=0, no salary cycle timing)
            # - Network error: Immediate retry (delay=0)
            # - 3DS timeout / Mandate expired / Card limit: Drop / No action
            # -------------------------------------------------------------
            cat = txn.failure_category
            if cat == FailureCategory.BANK_DOWNTIME:
                rule_action = InterventionType.SMART_RETRY_IMMEDIATE
                rule_delay = 0
            elif cat == FailureCategory.INSUFFICIENT_FUNDS:
                rule_action = InterventionType.SMS_PAY_LINK
                rule_delay = 0
            elif cat == FailureCategory.NETWORK_ERROR:
                rule_action = InterventionType.SMART_RETRY_IMMEDIATE
                rule_delay = 0
            elif cat == FailureCategory.MANDATE_EXPIRED:
                rule_action = InterventionType.SMART_RETRY_IMMEDIATE
                rule_delay = 0
            else:
                rule_action = InterventionType.NO_ACTION
                rule_delay = 0

            if rule_action != InterventionType.NO_ACTION:
                sim_rule = self.simulator_rule.simulate_outcome(
                    txn=txn,
                    action=rule_action,
                    delay_hours=rule_delay,
                )
                if sim_rule["is_recovered"]:
                    rule_results["gross_recovered"] += sim_rule["amount_recovered_inr"]
                    rule_results["recovered_count"] += 1
                rule_results["costs"] += sim_rule["operational_cost_inr"]
                rule_results["friction"] += sim_rule["friction_penalty_inr"]
                rule_results["interventions"] += 1

            # -------------------------------------------------------------
            # Policy 4: Naive Immediate 3x Retry
            # - Blindly executes 3 immediate retries on every single failure
            # -------------------------------------------------------------
            naive_results["costs"] += settings.COST_AUTO_RETRY_INR * 3.0
            naive_results["interventions"] += 3
            sim_naive = self.simulator_naive.simulate_outcome(
                txn=txn,
                action=InterventionType.SMART_RETRY_IMMEDIATE,
                delay_hours=0,
            )
            if sim_naive["is_recovered"]:
                naive_results["gross_recovered"] += txn.amount_inr
                naive_results["recovered_count"] += 1

        # Compute Net Returns
        recov_net = recov_results["gross_recovered"] - recov_results["costs"] - recov_results["friction"]
        noml_net = noml_results["gross_recovered"] - noml_results["costs"] - noml_results["friction"]
        rule_net = rule_results["gross_recovered"] - rule_results["costs"] - rule_results["friction"]
        naive_net = naive_results["gross_recovered"] - naive_results["costs"] - naive_results["friction"]

        recov_rate = (recov_results["gross_recovered"] / total_at_risk * 100.0) if total_at_risk > 0 else 0.0
        noml_rate = (noml_results["gross_recovered"] / total_at_risk * 100.0) if total_at_risk > 0 else 0.0
        rule_rate = (rule_results["gross_recovered"] / total_at_risk * 100.0) if total_at_risk > 0 else 0.0
        naive_rate = (naive_results["gross_recovered"] / total_at_risk * 100.0) if total_at_risk > 0 else 0.0

        summary = {
            "total_transactions": total_count,
            "total_revenue_at_risk_inr": round(total_at_risk, 2),
            "policies": {
                "RecoverAI": {
                    "policy_name": "RecoverAI (Full Agent with Calibrated ML)",
                    "recovered_count": recov_results["recovered_count"],
                    "recovery_rate_pct": round(recov_rate, 2),
                    "gross_recovered_inr": round(recov_results["gross_recovered"], 2),
                    "operational_costs_inr": round(recov_results["costs"], 2),
                    "friction_penalties_inr": round(recov_results["friction"], 2),
                    "net_revenue_recovered_inr": round(recov_net, 2),
                    "interventions_triggered": recov_results["interventions"],
                },
                "RecoverAI_Actions_No_ML": {
                    "policy_name": "RecoverAI_Actions_No_ML (Ablation: Policy Rules, Uniform Prior)",
                    "recovered_count": noml_results["recovered_count"],
                    "recovery_rate_pct": round(noml_rate, 2),
                    "gross_recovered_inr": round(noml_results["gross_recovered"], 2),
                    "operational_costs_inr": round(noml_results["costs"], 2),
                    "friction_penalties_inr": round(noml_results["friction"], 2),
                    "net_revenue_recovered_inr": round(noml_net, 2),
                    "interventions_triggered": noml_results["interventions"],
                },
                "Rule_Based_Heuristic": {
                    "policy_name": "Rule-Based Heuristic (Standard Dunning)",
                    "recovered_count": rule_results["recovered_count"],
                    "recovery_rate_pct": round(rule_rate, 2),
                    "gross_recovered_inr": round(rule_results["gross_recovered"], 2),
                    "operational_costs_inr": round(rule_results["costs"], 2),
                    "friction_penalties_inr": round(rule_results["friction"], 2),
                    "net_revenue_recovered_inr": round(rule_net, 2),
                    "interventions_triggered": rule_results["interventions"],
                },
                "Naive_Immediate_3x": {
                    "policy_name": "Naive Immediate 3x Retry",
                    "recovered_count": naive_results["recovered_count"],
                    "recovery_rate_pct": round(naive_rate, 2),
                    "gross_recovered_inr": round(naive_results["gross_recovered"], 2),
                    "operational_costs_inr": round(naive_results["costs"], 2),
                    "friction_penalties_inr": round(naive_results["friction"], 2),
                    "net_revenue_recovered_inr": round(naive_net, 2),
                    "interventions_triggered": naive_results["interventions"],
                },
                "No_Recovery": {
                    "policy_name": "No Intervention (Baseline)",
                    "recovered_count": 0,
                    "recovery_rate_pct": 0.0,
                    "gross_recovered_inr": 0.0,
                    "operational_costs_inr": 0.0,
                    "friction_penalties_inr": 0.0,
                    "net_revenue_recovered_inr": 0.0,
                    "interventions_triggered": 0,
                },
            },
            "comparison": {
                "net_lift_vs_rule_based_inr": round(recov_net - rule_net, 2),
                "net_lift_vs_naive_inr": round(recov_net - naive_net, 2),
                "macro_action_lift_vs_rules_inr": round(noml_net - rule_net, 2),
                "precision_ml_lift_vs_noml_inr": round(recov_net - noml_net, 2),
                "action_design_lift_vs_rule_based_inr": round(noml_net - rule_net, 2),
                "action_design_lift_vs_naive_inr": round(noml_net - naive_net, 2),
                "rate_lift_vs_rule_based_pct": round(recov_rate - rule_rate, 2),
                "rate_lift_vs_naive_pct": round(recov_rate - naive_rate, 2),
                "cost_savings_vs_naive_inr": round(naive_results["costs"] - recov_results["costs"], 2),
            },
        }

        return summary
