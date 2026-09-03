"""
Training script for RecoverAI Machine Learning Models.

Runs leak-free feature extraction, fits baseline models and CalibratedRecoveryClassifier,
evaluates on held-out test split, and logs both ML technical metrics and Physical Policy Benchmark.
"""

import logging
import pandas as pd
from recoverai.config.settings import settings
from recoverai.data.generator import generate_and_save_data
from recoverai.data.schemas import PaymentTransaction
from recoverai.features.pipeline import FeaturePipeline
from recoverai.models.baseline import RuleBasedBaseline, LogisticRegressionBaseline
from recoverai.models.recovery_classifier import CalibratedRecoveryClassifier
from recoverai.models.evaluator import ModelEvaluator
from recoverai.simulator.counterfactual import CounterfactualBenchmarkRunner

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("recoverai.models.train")


def train_and_evaluate_pipeline():
    """Executes end-to-end data generation, feature fitting, model training, and evaluation."""
    logger.info("Starting RecoverAI Model Training Pipeline...")
    
    # 1. Generate / Load Datasets
    train_df, val_df, test_df = generate_and_save_data()

    # 2. Feature Engineering & Scaling (Fitted ONLY on Train)
    logger.info("Fitting Feature Pipeline on Training set...")
    feature_pipeline = FeaturePipeline()
    X_train, y_train = feature_pipeline.fit_transform(train_df)
    X_val = feature_pipeline.transform(val_df)
    y_val = val_df["recovered_optimal"].astype(int)
    X_test = feature_pipeline.transform(test_df)
    y_test = test_df["recovered_optimal"].astype(int)
    amounts_test = test_df["amount_inr"]

    logger.info(f"Dataset split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

    # 3. Train Baseline Models
    logger.info("Training Logistic Regression Baseline...")
    lr_baseline = LogisticRegressionBaseline()
    lr_baseline.fit(X_train, y_train)
    lr_probs = lr_baseline.predict_proba(X_test)[:, 1]

    rule_baseline = RuleBasedBaseline()
    rule_probs = rule_baseline.predict_proba(X_test)[:, 1]

    # 4. Train Calibrated Recovery Classifier
    logger.info("Training Calibrated Recovery Classifier (LightGBM + Platt Calibration)...")
    classifier = CalibratedRecoveryClassifier()
    classifier.fit(X_train, y_train, X_val, y_val)
    ml_probs = classifier.predict_recovery_probability(X_test)

    # 5. Evaluate ML Probability Models Side-by-Side (Technical ML Metrics)
    models_dict = {
        "Rule-Based Heuristic": rule_probs,
        "Logistic Regression Baseline": lr_probs,
        "RecoverAI Calibrated Classifier": ml_probs,
    }

    comparison_df = ModelEvaluator.compare_models(
        y_true=y_test,
        amounts_inr=amounts_test,
        models_dict=models_dict,
    )

    logger.info("\n" + "=" * 80 + "\n1. ML CLASSIFICATION & PROBABILITY CALIBRATION BENCHMARK (HELD-OUT TEST SET)\n" + "=" * 80)
    print("\n" + comparison_df.to_string(index=False) + "\n")

    # 6. Physical Counterfactual Policy Benchmark
    logger.info("\n" + "=" * 80 + "\n2. PHYSICAL SIMULATION POLICY BENCHMARK (1,500 HELD-OUT TEST TRANSACTIONS)\n" + "=" * 80)
    clean_test_df = test_df.where(pd.notnull(test_df), None)
    txns = [PaymentTransaction(**r) for r in clean_test_df.to_dict(orient="records")]
    
    runner = CounterfactualBenchmarkRunner()
    benchmark_res = runner.run_benchmark(txns)

    sim_rows = []
    for pol_key, pol in benchmark_res["policies"].items():
        total_ded = pol["operational_costs_inr"] + pol["friction_penalties_inr"]
        sim_rows.append({
            "Policy": pol["policy_name"],
            "Recovered Count": f"{pol['recovered_count']} / {benchmark_res['total_transactions']}",
            "Recovery Rate (%)": f"{pol['recovery_rate_pct']:.1f}%",
            "Gross Recovered (INR)": f"INR {pol['gross_recovered_inr']:,.2f}",
            "Operational Costs (INR)": f"INR {pol['operational_costs_inr']:,.2f}",
            "Friction Costs (INR)": f"INR {pol['friction_penalties_inr']:,.2f}",
            "Total Deductions (INR)": f"INR {total_ded:,.2f}",
            "Net Revenue Recovered (INR)": f"INR {pol['net_revenue_recovered_inr']:,.2f}",
            "Interventions": pol["interventions_triggered"],
        })
    
    sim_df = pd.DataFrame(sim_rows)
    print("\n" + sim_df.to_string(index=False) + "\n")

    # Feature Importance
    importances = classifier.get_feature_importances()
    logger.info("Top 7 Most Predictive Features:")
    for feat, score in list(importances.items())[:7]:
        logger.info(f"  - {feat}: {score:.4f}")

    logger.info("ML Training & Evaluation successfully completed!")
    return comparison_df, benchmark_res


if __name__ == "__main__":
    train_and_evaluate_pipeline()
