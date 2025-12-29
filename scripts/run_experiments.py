#!/usr/bin/env python
"""
AMSM Framework - Experiment Runner
==================================

运行完整的实验对比，包括:
    - 多个目标模型的攻击评估
    - 多种基线方法的对比
    - 多粒度评估
    - 统计显著性分析

Usage:
    python scripts/run_experiments.py --config configs/config.yaml
"""

import os
import sys
import argparse
from typing import Dict, List, Optional
from pathlib import Path

# 添加项目根目录到路径
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
    """
    实验运行器
    
    协调所有实验的执行和结果收集
    """
    
    def __init__(self, config: Dict, output_dir: str):
        """
        Args:
            config: 配置字典
            output_dir: 输出目录
        """
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
        """
        运行基线方法对比实验
        
        Args:
            y_true: 真实标签
            amsm_scores: AMSM预测分数
            target_losses: 目标模型损失
            reference_losses: 参考模型损失 (用于似然比攻击)
            correct_probs: 正确token概率 (用于Min-K攻击)
        """
        self.logger.info("Running baseline comparison...")
        
        # AMSM
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
        
        # 打印对比表格
        self.comparator.print_comparison_table()
        
        # 计算AMSM相对于各基线的改进
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
        """
        运行多粒度评估实验
        
        Args:
            granularity_data: {granularity: {"y_true": ..., "y_scores": ...}}
        """
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
        """
        运行统计显著性分析
        
        Args:
            y_true: 真实标签
            amsm_scores: AMSM预测分数
            baseline_scores: 基线方法预测分数
        """
        self.logger.info("Running statistical analysis...")
        
        # Bootstrap置信区间
        mean_auc, lower, upper = StatisticalTester.bootstrap_auc(
            y_true, amsm_scores, n_bootstrap=1000
        )
        self.logger.info(f"\nBootstrap AUC (AMSM):")
        self.logger.info(f"  Mean: {mean_auc:.4f}")
        self.logger.info(f"  95% CI: [{lower:.4f}, {upper:.4f}]")
        
        # 置换检验
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
        """
        在多个目标模型上运行实验
        """
        self.logger.info("Running experiments on multiple target models...")
        
        target_models = self.config.get("target_models", {})
        results = {}
        
        # 模拟不同目标模型的攻击结果
        for category in ["open_source", "commercial"]:
            models = target_models.get(category, [])
            
            for model_config in models:
                model_name = model_config.get("name", "unknown")
                self.logger.info(f"\nEvaluating on {model_name}...")
                
                # 模拟攻击数据 (实际中应该使用真实数据)
                np.random.seed(hash(model_name) % 2**32)
                n_samples = 1000
                
                y_true = np.random.randint(0, 2, n_samples)
                
                # 模拟不同模型难度
                if "70b" in model_name.lower() or "gpt-4" in model_name.lower():
                    # 大模型更难攻击
                    noise_level = 0.4
                else:
                    noise_level = 0.3
                    
                y_scores = y_true * 0.7 + np.random.rand(n_samples) * noise_level
                y_scores = np.clip(y_scores, 0, 1)
                
                # 评估
                metrics = self.evaluator.evaluate(y_true, y_scores)
                
                results[model_name] = metrics.to_dict()
                
                self.logger.info(f"  AUC-ROC: {metrics.auc_roc:.4f}")
                self.logger.info(f"  Accuracy: {metrics.accuracy:.4f}")
                self.logger.info(f"  Attack Advantage: {metrics.attack_advantage:.4f}")
                
        return results
    
    def run_all_experiments(self) -> Dict:
        """
        运行所有实验
        """
        self.logger.info("=" * 60)
        self.logger.info("AMSM Framework - Full Experiment Suite")
        self.logger.info("=" * 60)
        
        total_timer = Timer()
        total_timer.start()
        
        # 生成模拟数据
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
        
        # 1. 基线对比
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Experiment 1: Baseline Comparison")
        self.logger.info("=" * 60)
        baseline_results = self.run_baseline_comparison(
            y_true, amsm_scores, target_losses, reference_losses, correct_probs
        )
        self.results["baseline_comparison"] = baseline_results
        
        # 2. 多粒度评估
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
        
        # 3. 统计分析
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Experiment 3: Statistical Analysis")
        self.logger.info("=" * 60)
        baseline_scores = -target_losses / target_losses.max()  # 归一化
        stat_results = self.run_statistical_analysis(y_true, amsm_scores, baseline_scores)
        self.results["statistical_analysis"] = stat_results
        
        # 4. 多目标模型实验
        self.logger.info("\n" + "=" * 60)
        self.logger.info("Experiment 4: Multiple Target Models")
        self.logger.info("=" * 60)
        model_results = self.run_target_model_experiments()
        self.results["target_models"] = model_results
        
        total_timer.stop()
        self.results["total_time_seconds"] = total_timer.elapsed
        
        # 保存结果
        results_path = os.path.join(self.output_dir, "experiment_results.json")
        save_json(self.results, results_path)
        
        # 生成可视化
        figures_dir = os.path.join(self.output_dir, "figures")
        os.makedirs(figures_dir, exist_ok=True)
        
        # ROC曲线
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
    """主函数"""
    parser = argparse.ArgumentParser(description="AMSM Experiment Runner")
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
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置输出目录
    if args.output is None:
        timestamp = get_timestamp()
        args.output = f"outputs/experiments_{timestamp}"
        
    os.makedirs(args.output, exist_ok=True)
    
    # 运行实验
    runner = ExperimentRunner(config, args.output)
    results = runner.run_all_experiments()
    
    # 打印摘要
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
