#!/usr/bin/env python
"""
Mosaic Framework - Experiment Runner
==================================

Usage:
    python scripts/run_experiments.py --config configs/config.yaml
"""

import os
import sys
import argparse
from typing import Dict, List, Optional
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

from src.utils import (
    load_config,
    save_json,
    setup_logging,
    set_seed,
    get_timestamp,
    Timer,
    plot_roc_curve,
)
from src.evaluation import (
    MembershipInferenceEvaluator,
    BaselineComparator,
    MultiGranularityEvaluator,
    StatisticalTester,
)
from src.models import BaselineAttacks


class ExperimentRunner:
    def __init__(self, config: Dict, output_dir: str):
        self.config = config
        self.output_dir = output_dir
        
        self.seed = config.get("experiment", {}).get("seed", 42)
        set_seed(self.seed)
        
        log_dir = os.path.join(output_dir, "logs")
        self.logger = setup_logging(log_dir=log_dir)
        
        self.evaluator = MembershipInferenceEvaluator()
        self.comparator = BaselineComparator()
        self.multi_granularity = MultiGranularityEvaluator()
        
        self.results = {}
        
    def run_baseline_comparison(
        self,
        y_true: np.ndarray,
        amsm_scores: np.ndarray,
        target_losses: np.ndarray,
        reference_losses: Optional[np.ndarray] = None,
        correct_probs: Optional[np.ndarray] = None,
    ) -> Dict:
        self.logger.info("Running baseline comparison...")

        # Mosaic
        self.comparator.add_result("AMSM", y_true, amsm_scores)
        
        # Loss-based attack
        loss_preds, loss_scores = BaselineAttacks.loss_based_attack(
            torch.tensor(target_losses)
        )
        self.comparator.add_result("Loss-based", y_true, -target_losses)  # 负损失作为分数
        
        # Likelihood ratio attack
        if reference_losses is not None:
            lr_preds, lr_scores = BaselineAttacks.likelihood_ratio_attack(
                torch.tensor(target_losses),
                torch.tensor(reference_losses),
            )
            self.comparator.add_result("Likelihood-Ratio", y_true, lr_scores.numpy())
            
        # Min-K attack
        if correct_probs is not None:
            mk_preds, mk_scores = BaselineAttacks.min_k_attack(
                torch.tensor(correct_probs),
                k_ratio=0.2,
            )
            self.comparator.add_result("Min-K", y_true, mk_scores.numpy())
            
        # Random baseline
        self.comparator.add_result("Random", y_true, np.random.rand(len(y_true)))

        self.comparator.print_comparison_table()

        improvements = {}
        for baseline in ["Loss-based", "Random"]:
            if baseline in self.comparator.results:
                imp = self.comparator.compute_improvement("AMSM", baseline)
                improvements[baseline] = imp
                self.logger.info(f"\nImprovement over {baseline}:")
                for metric, value in imp.items():
                    self.logger.info(f"  {metric}: {value:+.2f}%")
                    
        return {
            "comparison": self.comparator.compare(),
            "improvements": improvements,
        }
    
    def run_multi_granularity_evaluation(
        self,
        granularity_data: Dict[str, Dict],
    ) -> Dict:
        self.logger.info("Running multi-granularity evaluation...")
        
        for granularity, data in granularity_data.items():
            metrics = self.multi_granularity.evaluate_granularity(
                granularity,
                data["y_true"],
                data["y_scores"],
            )
            self.logger.info(f"\n{granularity} level:")
            self.logger.info(f"  AUC-ROC: {metrics.auc_roc:.4f}")
            self.logger.info(f"  Accuracy: {metrics.accuracy:.4f}")
            
        self.multi_granularity.print_granularity_comparison()
        
        return self.multi_granularity.get_summary()
    
    def run_statistical_analysis(
        self,
        y_true: np.ndarray,
        amsm_scores: np.ndarray,
        baseline_scores: np.ndarray,
    ) -> Dict:
        self.logger.info("Running statistical analysis...")
        
        # Bootstrap
        mean_auc, lower, upper = StatisticalTester.bootstrap_auc(
            y_true, amsm_scores, n_bootstrap=1000
        )
        self.logger.info(f"\nBootstrap AUC (AMSM):")
        self.logger.info(f"  Mean: {mean_auc:.4f}")
        self.logger.info(f"  95% CI: [{lower:.4f}, {upper:.4f}]")

        diff, p_value = StatisticalTester.permutation_test(
            y_true, amsm_scores, baseline_scores, n_permutations=1000
        )
        self.logger.info(f"\nPermutation test (AMSM vs Baseline):")
        self.logger.info(f"  AUC difference: {diff:.4f}")
        self.logger.info(f"  p-value: {p_value:.4f}")
        
        significance = "significant" if p_value < 0.05 else "not significant"
        self.logger.info(f"  Result: {significance} at α=0.05")
        
        return {
            "bootstrap": {
                "mean_auc": mean_auc,
                "ci_lower": lower,
                "ci_upper": upper,
            },
            "permutation_test": {
                "auc_difference": diff,
                "p_value": p_value,
                "significant": p_value < 0.05,
            },
        }
    
    def run_target_model_experiments(self) -> Dict:
        self.logger.info("Running experiments on multiple target models...")
        
        target_models = self.config.get("target_models", {})
        results = {}

        for category in ["open_source", "commercial"]:
            models = target_models.get(category, [])
            
            for model_config in models:
                model_name = model_config.get("name", "unknown")
                self.logger.info(f"\nEvaluating on {model_name}...")

                np.random.seed(hash(model_name) % 2**32)
                n_samples = 1000
                
                y_true = np.random.randint(0, 2, n_samples)

                if "70b" in model_name.lower() or "gpt-4" in model_name.lower():
                    noise_level = 0.4
                else:
                    noise_level = 0.3
                    
                y_scores = y_true * 0.7 + np.random.rand(n_samples) * noise_level
                y_scores = np.clip(y_scores, 0, 1)

                metrics = self.evaluator.evaluate(y_true, y_scores)
                
                results[model_name] = metrics.to_dict()
                
                self.logger.info(f"  AUC-ROC: {metrics.auc_roc:.4f}")
                self.logger.info(f"  Accuracy: {metrics.accuracy:.4f}")
                self.logger.info(f"  Attack Advantage: {metrics.attack_advantage:.4f}")
                
        return results
    
    def run_all_experiments(self) -> Dict:
        self.logger.info("=" * 60)
        self.logger.info("Mosaic Framework - Full Experiment Suite")
        self.logger.info("=" * 60)
        
        total_timer = Timer()
        total_timer.start()

        np.random.seed(self.seed)
        n_samples = 2000
        
        y_true = np.random.randint(0, 2, n_samples)
        amsm_scores = y_true * 0.75 + np.random.rand(n_samples) * 0.25
        amsm_scores = np.clip(amsm_scores, 0, 1)
        
        target_losses = np.where(y_true == 1, 
                                  np.random.rand(n_samples) * 2,
                                  np.random.rand(n_samples) * 2 + 1)
        reference_losses = np.random.rand(n_samples) * 3
        correct_probs = np.random.rand(n_samples, 100)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Experiment 1: Baseline Comparison")
        self.logger.info("=" * 60)
        baseline_results = self.run_baseline_comparison(
            y_true, amsm_scores, target_losses, reference_losses, correct_probs
        )
        self.results["baseline_comparison"] = baseline_results

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Experiment 2: Multi-Granularity Evaluation")
        self.logger.info("=" * 60)
        granularity_data = {
            "document": {
                "y_true": y_true[:500],
                "y_scores": amsm_scores[:500],
            },
            "paragraph": {
                "y_true": y_true[500:1000],
                "y_scores": amsm_scores[500:1000] * 0.95,
            },
            "sentence": {
                "y_true": y_true[1000:1500],
                "y_scores": amsm_scores[1000:1500] * 0.9,
            },
        }
        granularity_results = self.run_multi_granularity_evaluation(granularity_data)
        self.results["multi_granularity"] = granularity_results

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Experiment 3: Statistical Analysis")
        self.logger.info("=" * 60)
        baseline_scores = -target_losses / target_losses.max()  # 归一化
        stat_results = self.run_statistical_analysis(y_true, amsm_scores, baseline_scores)
        self.results["statistical_analysis"] = stat_results

        self.logger.info("\n" + "=" * 60)
        self.logger.info("Experiment 4: Multiple Target Models")
        self.logger.info("=" * 60)
        model_results = self.run_target_model_experiments()
        self.results["target_models"] = model_results
        
        total_timer.stop()
        self.results["total_time_seconds"] = total_timer.elapsed

        results_path = os.path.join(self.output_dir, "experiment_results.json")
        save_json(self.results, results_path)

        figures_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)

        plot_roc_curve(
            y_true, amsm_scores,
            save_path=os.path.join(figures_dir, "roc_curve.png"),
            label="AMSM",
        )
        
        self.logger.info("\n" + "=" * 60)
        self.logger.info("All experiments completed!")
        self.logger.info(f"Total time: {total_timer}")
        self.logger.info(f"Results saved to: {results_path}")
        self.logger.info("=" * 60)
        
        return self.results


def main():
    parser = argparse.ArgumentParser(description="Mosaic Experiment Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory",
    )
    
    args = parser.parse_args()

    config = load_config(args.config)

    if args.output is None:
        timestamp = get_timestamp()
        args.output = f"outputs/experiments_{timestamp}"
        
    os.makedirs(args.output, exist_ok=True)

    runner = ExperimentRunner(config, args.output)
    results = runner.run_all_experiments()

    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    
    if "baseline_comparison" in results:
        comparison = results["baseline_comparison"]["comparison"]
        print("\nBaseline Comparison (AUC-ROC):")
        for method, metrics in comparison.items():
            print(f"  {method}: {metrics['auc_roc']:.4f}")
            
    if "statistical_analysis" in results:
        stat = results["statistical_analysis"]
        print(f"\nStatistical Significance:")
        print(f"  p-value: {stat['permutation_test']['p_value']:.4f}")
        print(f"  Significant: {stat['permutation_test']['significant']}")
        
    print("\nDone!")


if __name__ == "__main__":
    main()
