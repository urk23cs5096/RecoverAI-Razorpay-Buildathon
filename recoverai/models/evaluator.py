"""
Comprehensive Model & Business Evaluation Engine for RecoverAI.

Calculates technical ML metrics (ROC-AUC, PR-AUC, Calibration, Brier Score)
alongside financial business impact metrics (Net Recovered Revenue, Lift, Cost Savings).
"""

import logging
from typing import Dict, Any, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve

from recoverai.config.settings import settings

logger = logging.getLogger("recoverai.models.evaluator")


class ModelEvaluator:
    """Evaluates ML recovery classifiers across technical metrics and merchant business ROI."""

    @staticmethod
    def calculate_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Computes Expected Calibration Error (ECE)."""
        bin_limits = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n = len(y_true)
        
        for i in range(n_bins):
            bin_mask = (y_prob >= bin_limits[i]) & (y_prob < bin_limits[i + 1])
            if np.sum(bin_mask) > 0:
                bin_acc = np.mean(y_true[bin_mask])
                bin_conf = np.mean(y_prob[bin_mask])
                ece += (np.sum(bin_mask) / n) * np.abs(bin_acc - bin_conf)
        return float(ece)

    @classmethod
    def evaluate_model(
        cls,
        model_name: str,
        y_true: pd.Series,
        y_prob: np.ndarray,
        amounts_inr: pd.Series,
        threshold: float = 0.50,
        intervention_cost_inr: float = 0.80,
    ) -> Dict[str, Any]:
        """
        Computes full technical and business evaluation suite.
        """
        y_true_arr = np.array(y_true).astype(int)
        y_pred = (y_prob >= threshold).astype(int)
        amounts_arr = np.array(amounts_inr)

        # 1. Technical ML Metrics
        roc_auc = float(roc_auc_score(y_true_arr, y_prob))
        pr_auc = float(average_precision_score(y_true_arr, y_prob))
        precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
        recall = float(recall_score(y_true_arr, y_pred, zero_division=0))
        f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))
        brier = float(brier_score_loss(y_true_arr, y_prob))
        ece = cls.calculate_ece(y_true_arr, y_prob)

        # Confusion Matrix
        cm = confusion_matrix(y_true_arr, y_pred)
        tn, fp, fn, tp = cm.ravel()

        # Calibration Curve Points
        prob_true, prob_pred = calibration_curve(y_true_arr, y_prob, n_bins=10, strategy="uniform")

        # 2. Financial & Business ROI Metrics
        # When model predicts positive (1): we trigger intervention costing `intervention_cost_inr`
        # If true positive: we recover the transaction amount
        # If false positive: we waste the intervention cost + friction penalty
        # If false negative: we lose potential recoverable amount
        # If true negative: we saved intervention cost on hopeless transaction
        
        tp_mask = (y_pred == 1) & (y_true_arr == 1)
        fp_mask = (y_pred == 1) & (y_true_arr == 0)
        fn_mask = (y_pred == 0) & (y_true_arr == 1)
        tn_mask = (y_pred == 0) & (y_true_arr == 0)

        gross_recovered_inr = float(np.sum(amounts_arr[tp_mask]))
        missed_revenue_inr = float(np.sum(amounts_arr[fn_mask]))
        total_at_risk_inr = float(np.sum(amounts_arr))
        
        interventions_triggered = int(tp + fp)
        total_intervention_costs_inr = interventions_triggered * intervention_cost_inr
        friction_penalties_inr = int(fp) * settings.CUSTOMER_FRICTION_PENALTY_INR
        
        net_recovered_inr = gross_recovered_inr - total_intervention_costs_inr - friction_penalties_inr
        recovery_rate = (gross_recovered_inr / total_at_risk_inr) * 100.0 if total_at_risk_inr > 0 else 0.0

        return {
            "model_name": model_name,
            "threshold": threshold,
            "sample_count": len(y_true_arr),
            # ML Metrics
            "roc_auc": round(roc_auc, 4),
            "pr_auc": round(pr_auc, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "brier_score": round(brier, 4),
            "ece": round(ece, 4),
            # Confusion Matrix Counts
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
            # Financial Business Metrics (in INR)
            "total_revenue_at_risk_inr": round(total_at_risk_inr, 2),
            "gross_revenue_recovered_inr": round(gross_recovered_inr, 2),
            "total_intervention_costs_inr": round(total_intervention_costs_inr, 2),
            "friction_penalties_inr": round(friction_penalties_inr, 2),
            "net_recovered_inr": round(net_recovered_inr, 2),
            "missed_revenue_inr": round(missed_revenue_inr, 2),
            "recovery_rate_pct": round(recovery_rate, 2),
            "interventions_triggered": interventions_triggered,
            # Calibration plot coordinates
            "calibration_curve": {
                "prob_true": [round(float(p), 4) for p in prob_true],
                "prob_pred": [round(float(p), 4) for p in prob_pred],
            }
        }

    @classmethod
    def compare_models(
        cls,
        y_true: pd.Series,
        amounts_inr: pd.Series,
        models_dict: Dict[str, np.ndarray],
    ) -> pd.DataFrame:
        """Compares multiple models side-by-side in a summary DataFrame."""
        rows = []
        for name, probs in models_dict.items():
            metrics = cls.evaluate_model(
                model_name=name,
                y_true=y_true,
                y_prob=probs,
                amounts_inr=amounts_inr,
            )
            rows.append({
                "Model": name,
                "ROC-AUC": metrics["roc_auc"],
                "PR-AUC": metrics["pr_auc"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "F1": metrics["f1_score"],
                "Brier Score": metrics["brier_score"],
                "ECE": metrics["ece"],
                "Gross Recovered (INR)": f"INR {metrics['gross_revenue_recovered_inr']:,.2f}",
                "Net Recovered (INR)": f"INR {metrics['net_recovered_inr']:,.2f}",
                "Recovery Rate (%)": f"{metrics['recovery_rate_pct']:.1f}%",
                "Interventions": metrics["interventions_triggered"],
            })
        return pd.DataFrame(rows)
