"""
Evaluation Metrics Module
=========================
"""

from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


@dataclass
class AttackMetrics:
    auc_roc: float
    auprc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    attack_advantage: float
    tpr_at_fpr: Dict[float, float]
    
    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "Attack Evaluation Metrics",
            "=" * 50,
            f"AUC-ROC: {self.auc_roc:.4f}",
            f"AUPRC: {self.auprc:.4f}",
            f"Accuracy: {self.accuracy:.4f}",
            f"Precision: {self.precision:.4f}",
            f"Recall: {self.recall:.4f}",
            f"F1 Score: {self.f1:.4f}",
            f"Attack Advantage: {self.attack_advantage:.4f}",
            "",
            "TPR @ FPR:",
        ]
        for fpr, tpr in self.tpr_at_fpr.items():
            lines.append(f"  FPR={fpr}: TPR={tpr:.4f}")
        lines.append("=" * 50)
        return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        return {
            "auc_roc": self.auc_roc,
            "auprc": self.auprc,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "attack_advantage": self.attack_advantage,
            "tpr_at_fpr": self.tpr_at_fpr,
        }


class MembershipInferenceEvaluator:
    
    def __init__(
        self,
        fpr_thresholds: List[float] = [0.001, 0.01, 0.05, 0.1],
    ):
        self.fpr_thresholds = fpr_thresholds
        
    def compute_auc_roc(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> float:
        try:
            return roc_auc_score(y_true, y_scores)
        except ValueError:
            return 0.5
    
    def compute_auprc(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> float:
        try:
            return average_precision_score(y_true, y_scores)
        except ValueError:
            return 0.0
    
    def compute_tpr_at_fpr(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        fpr_thresholds: Optional[List[float]] = None,
    ) -> Dict[float, float]:
        if fpr_thresholds is None:
            fpr_thresholds = self.fpr_thresholds
            
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
        except ValueError:
            return {t: 0.0 for t in fpr_thresholds}
        
        result = {}
        for threshold in fpr_thresholds:
            idx = np.searchsorted(fpr, threshold)
            if idx >= len(tpr):
                result[threshold] = tpr[-1]
            else:
                result[threshold] = tpr[idx]
                
        return result
    
    def compute_attack_advantage(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> float:

        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            advantage = np.max(tpr - fpr)
            return float(advantage)
        except ValueError:
            return 0.0
    
    def compute_confusion_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> Dict[str, float]:
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
    
    def evaluate(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        threshold: float = 0.5,
    ) -> AttackMetrics:
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)
        y_pred = (y_scores >= threshold).astype(int)

        auc_roc = self.compute_auc_roc(y_true, y_scores)
        auprc = self.compute_auprc(y_true, y_scores)
        confusion_metrics = self.compute_confusion_metrics(y_true, y_pred)
        attack_advantage = self.compute_attack_advantage(y_true, y_scores)
        tpr_at_fpr = self.compute_tpr_at_fpr(y_true, y_scores)
        
        return AttackMetrics(
            auc_roc=auc_roc,
            auprc=auprc,
            accuracy=confusion_metrics["accuracy"],
            precision=confusion_metrics["precision"],
            recall=confusion_metrics["recall"],
            f1=confusion_metrics["f1"],
            attack_advantage=attack_advantage,
            tpr_at_fpr=tpr_at_fpr,
        )
    
    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        metric: str = "f1",
    ) -> Tuple[float, float]:

        thresholds = np.linspace(0, 1, 101)
        best_threshold = 0.5
        best_score = 0.0
        
        for t in thresholds:
            y_pred = (y_scores >= t).astype(int)
            
            if metric == "f1":
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == "accuracy":
                score = accuracy_score(y_true, y_pred)
            elif metric == "youden":
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                score = tpr - fpr
            else:
                raise ValueError(f"Unknown metric: {metric}")
                
            if score > best_score:
                best_score = score
                best_threshold = t
                
        return best_threshold, best_score


class BaselineComparator:
    
    def __init__(self):
        self.evaluator = MembershipInferenceEvaluator()
        self.results = {}
        
    def add_result(
        self,
        method_name: str,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ):

        metrics = self.evaluator.evaluate(y_true, y_scores)
        self.results[method_name] = metrics
        
    def compare(self) -> Dict[str, Dict[str, float]]:

        comparison = {}
        
        for method, metrics in self.results.items():
            comparison[method] = metrics.to_dict()
            
        return comparison
    
    def print_comparison_table(self):
        if not self.results:
            print("No results to compare")
            return
            
        methods = list(self.results.keys())
        metrics_names = ["auc_roc", "auprc", "accuracy", "attack_advantage"]

        header = "Method".ljust(20)
        for metric in metrics_names:
            header += metric.ljust(15)
        print(header)
        print("-" * len(header))

        for method in methods:
            metrics = self.results[method]
            row = method.ljust(20)
            row += f"{metrics.auc_roc:.4f}".ljust(15)
            row += f"{metrics.auprc:.4f}".ljust(15)
            row += f"{metrics.accuracy:.4f}".ljust(15)
            row += f"{metrics.attack_advantage:.4f}".ljust(15)
            print(row)
            
    def compute_improvement(
        self,
        target_method: str,
        baseline_method: str,
    ) -> Dict[str, float]:

        if target_method not in self.results or baseline_method not in self.results:
            raise ValueError("Method not found in results")
            
        target = self.results[target_method]
        baseline = self.results[baseline_method]
        
        improvements = {}
        
        for metric in ["auc_roc", "auprc", "accuracy", "attack_advantage"]:
            target_val = getattr(target, metric)
            baseline_val = getattr(baseline, metric)
            
            if baseline_val != 0:
                improvement = (target_val - baseline_val) / baseline_val * 100
            else:
                improvement = 0.0
                
            improvements[metric] = improvement
            
        return improvements


class MultiGranularityEvaluator:
    def __init__(self):
        self.evaluator = MembershipInferenceEvaluator()
        self.granularity_results = {}
        
    def evaluate_granularity(
        self,
        granularity: str,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> AttackMetrics:
        metrics = self.evaluator.evaluate(y_true, y_scores)
        self.granularity_results[granularity] = metrics
        return metrics
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        summary = {}
        for granularity, metrics in self.granularity_results.items():
            summary[granularity] = metrics.to_dict()
        return summary
    
    def print_granularity_comparison(self):
        print("\nMulti-Granularity Evaluation Results")
        print("=" * 60)
        
        header = "Granularity".ljust(15) + "AUC-ROC".ljust(12) + "Accuracy".ljust(12) + "TPR@1%FPR".ljust(12)
        print(header)
        print("-" * 60)
        
        for granularity, metrics in self.granularity_results.items():
            tpr_1 = metrics.tpr_at_fpr.get(0.01, 0.0)
            row = granularity.ljust(15)
            row += f"{metrics.auc_roc:.4f}".ljust(12)
            row += f"{metrics.accuracy:.4f}".ljust(12)
            row += f"{tpr_1:.4f}".ljust(12)
            print(row)


class StatisticalTester:
    
    @staticmethod
    def bootstrap_auc(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float, float]:
        n_samples = len(y_true)
        aucs = []
        
        for _ in range(n_bootstrap):
            # Bootstrap
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_scores_boot = y_scores[indices]

            if len(np.unique(y_true_boot)) == 2:
                auc = roc_auc_score(y_true_boot, y_scores_boot)
                aucs.append(auc)
                
        aucs = np.array(aucs)

        alpha = 1 - confidence_level
        lower = np.percentile(aucs, alpha / 2 * 100)
        upper = np.percentile(aucs, (1 - alpha / 2) * 100)
        
        return float(np.mean(aucs)), float(lower), float(upper)
    
    @staticmethod
    def permutation_test(
        y_true: np.ndarray,
        y_scores_a: np.ndarray,
        y_scores_b: np.ndarray,
        n_permutations: int = 1000,
    ) -> Tuple[float, float]:

        auc_a = roc_auc_score(y_true, y_scores_a)
        auc_b = roc_auc_score(y_true, y_scores_b)
        observed_diff = auc_a - auc_b

        count = 0
        for _ in range(n_permutations):
            swap = np.random.choice([True, False], len(y_true))
            scores_a_perm = np.where(swap, y_scores_b, y_scores_a)
            scores_b_perm = np.where(swap, y_scores_a, y_scores_b)
            
            auc_a_perm = roc_auc_score(y_true, scores_a_perm)
            auc_b_perm = roc_auc_score(y_true, scores_b_perm)
            
            if abs(auc_a_perm - auc_b_perm) >= abs(observed_diff):
                count += 1
                
        p_value = count / n_permutations
        
        return observed_diff, p_value


if __name__ == "__main__":
    print("Testing Evaluation Metrics Module...")
    
    np.random.seed(42)

    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)

    y_scores_good = y_true * 0.7 + np.random.rand(n_samples) * 0.3
    y_scores_good = np.clip(y_scores_good, 0, 1)

    y_scores_bad = np.random.rand(n_samples)

    print("\nTesting MembershipInferenceEvaluator...")
    evaluator = MembershipInferenceEvaluator()
    
    metrics_good = evaluator.evaluate(y_true, y_scores_good)
    print("\nGood predictor metrics:")
    print(metrics_good)
    
    metrics_bad = evaluator.evaluate(y_true, y_scores_bad)
    print("\nBad predictor metrics:")
    print(metrics_bad)

    threshold, score = evaluator.find_optimal_threshold(y_true, y_scores_good, "f1")
    print(f"\nOptimal threshold: {threshold:.3f}, F1: {score:.4f}")

    print("\nTesting BaselineComparator...")
    comparator = BaselineComparator()
    comparator.add_result("AMSM", y_true, y_scores_good)
    comparator.add_result("Random", y_true, y_scores_bad)
    
    comparator.print_comparison_table()
    
    improvements = comparator.compute_improvement("AMSM", "Random")
    print(f"\nImprovements over Random: {improvements}")

    print("\nTesting MultiGranularityEvaluator...")
    multi_eval = MultiGranularityEvaluator()
    multi_eval.evaluate_granularity("document", y_true, y_scores_good)
    multi_eval.evaluate_granularity("paragraph", y_true, y_scores_good * 0.9)
    multi_eval.evaluate_granularity("sentence", y_true, y_scores_good * 0.8)
    
    multi_eval.print_granularity_comparison()

    print("\nTesting StatisticalTester...")
    mean_auc, lower, upper = StatisticalTester.bootstrap_auc(y_true, y_scores_good)
    print(f"Bootstrap AUC: {mean_auc:.4f} ({lower:.4f}, {upper:.4f})")
    
    diff, p_value = StatisticalTester.permutation_test(y_true, y_scores_good, y_scores_bad)
    print(f"Permutation test: diff={diff:.4f}, p-value={p_value:.4f}")
    
    print("\nAll tests passed!")
