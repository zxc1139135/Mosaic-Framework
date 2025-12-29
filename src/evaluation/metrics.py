"""
Evaluation Metrics Module
=========================

成员推理攻击评估指标模块。

论文参考:
    - 阶段IV: 攻击执行与评估 (Section 5.3)
    - 评估指标: AUC-ROC, TPR@FPR, Attack Advantage, AUPRC
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
    """攻击评估指标结果"""
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
    """
    成员推理攻击评估器
    
    提供完整的攻击性能评估功能
    """
    
    def __init__(
        self,
        fpr_thresholds: List[float] = [0.001, 0.01, 0.05, 0.1],
    ):
        """
        Args:
            fpr_thresholds: TPR@FPR评估的FPR阈值列表
        """
        self.fpr_thresholds = fpr_thresholds
        
    def compute_auc_roc(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> float:
        """
        计算AUC-ROC
        
        论文: AUC-ROC作为综合指标
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            
        Returns:
            auc: AUC-ROC值
        """
        try:
            return roc_auc_score(y_true, y_scores)
        except ValueError:
            return 0.5
    
    def compute_auprc(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> float:
        """
        计算AUPRC (Area Under Precision-Recall Curve)
        
        论文: AUPRC关注精确率
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            
        Returns:
            auprc: AUPRC值
        """
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
        """
        计算在给定FPR约束下的TPR
        
        论文: TPR@FPR衡量固定假阳性率约束下的真阳性率
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            fpr_thresholds: FPR阈值列表
            
        Returns:
            tpr_dict: {fpr_threshold: tpr}
        """
        if fpr_thresholds is None:
            fpr_thresholds = self.fpr_thresholds
            
        try:
            fpr, tpr, _ = roc_curve(y_true, y_scores)
        except ValueError:
            return {t: 0.0 for t in fpr_thresholds}
        
        result = {}
        for threshold in fpr_thresholds:
            # 找到最接近目标FPR的点
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
        """
        计算攻击优势 (Attack Advantage)
        
        论文: 攻击优势衡量最大可区分度
        
        定义: max(TPR - FPR) over all thresholds
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            
        Returns:
            advantage: 攻击优势值
        """
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
        """
        计算混淆矩阵相关指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            
        Returns:
            metrics: 指标字典
        """
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
        """
        完整评估
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数 (概率)
            threshold: 决策阈值
            
        Returns:
            metrics: 评估指标对象
        """
        y_true = np.asarray(y_true)
        y_scores = np.asarray(y_scores)
        y_pred = (y_scores >= threshold).astype(int)
        
        # 计算各项指标
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
        """
        寻找最优决策阈值
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            metric: 优化目标 ("f1", "accuracy", "youden")
            
        Returns:
            threshold: 最优阈值
            score: 对应指标值
        """
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
                # Youden's J statistic
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
    """
    基线方法对比器
    
    对比AMSM与传统攻击方法的性能
    """
    
    def __init__(self):
        self.evaluator = MembershipInferenceEvaluator()
        self.results = {}
        
    def add_result(
        self,
        method_name: str,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ):
        """
        添加一个方法的结果
        
        Args:
            method_name: 方法名称
            y_true: 真实标签
            y_scores: 预测分数
        """
        metrics = self.evaluator.evaluate(y_true, y_scores)
        self.results[method_name] = metrics
        
    def compare(self) -> Dict[str, Dict[str, float]]:
        """
        对比所有方法
        
        Returns:
            comparison: 对比结果字典
        """
        comparison = {}
        
        for method, metrics in self.results.items():
            comparison[method] = metrics.to_dict()
            
        return comparison
    
    def print_comparison_table(self):
        """打印对比表格"""
        if not self.results:
            print("No results to compare")
            return
            
        methods = list(self.results.keys())
        metrics_names = ["auc_roc", "auprc", "accuracy", "attack_advantage"]
        
        # 打印表头
        header = "Method".ljust(20)
        for metric in metrics_names:
            header += metric.ljust(15)
        print(header)
        print("-" * len(header))
        
        # 打印每个方法的结果
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
        """
        计算相对于基线的改进
        
        Args:
            target_method: 目标方法
            baseline_method: 基线方法
            
        Returns:
            improvements: 各指标的改进百分比
        """
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
    """
    多粒度评估器
    
    在文档级、段落级、句子级三个粒度上评估攻击性能
    """
    
    def __init__(self):
        self.evaluator = MembershipInferenceEvaluator()
        self.granularity_results = {}
        
    def evaluate_granularity(
        self,
        granularity: str,
        y_true: np.ndarray,
        y_scores: np.ndarray,
    ) -> AttackMetrics:
        """
        评估特定粒度
        
        Args:
            granularity: 粒度 ("document", "paragraph", "sentence")
            y_true: 真实标签
            y_scores: 预测分数
            
        Returns:
            metrics: 评估指标
        """
        metrics = self.evaluator.evaluate(y_true, y_scores)
        self.granularity_results[granularity] = metrics
        return metrics
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """获取所有粒度的摘要"""
        summary = {}
        for granularity, metrics in self.granularity_results.items():
            summary[granularity] = metrics.to_dict()
        return summary
    
    def print_granularity_comparison(self):
        """打印粒度对比"""
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
    """
    统计显著性测试
    
    评估攻击性能的统计显著性
    """
    
    @staticmethod
    def bootstrap_auc(
        y_true: np.ndarray,
        y_scores: np.ndarray,
        n_bootstrap: int = 1000,
        confidence_level: float = 0.95,
    ) -> Tuple[float, float, float]:
        """
        Bootstrap AUC置信区间估计
        
        Args:
            y_true: 真实标签
            y_scores: 预测分数
            n_bootstrap: Bootstrap采样次数
            confidence_level: 置信水平
            
        Returns:
            mean_auc: 平均AUC
            lower: 置信区间下界
            upper: 置信区间上界
        """
        n_samples = len(y_true)
        aucs = []
        
        for _ in range(n_bootstrap):
            # Bootstrap采样
            indices = np.random.choice(n_samples, n_samples, replace=True)
            y_true_boot = y_true[indices]
            y_scores_boot = y_scores[indices]
            
            # 确保有两个类别
            if len(np.unique(y_true_boot)) == 2:
                auc = roc_auc_score(y_true_boot, y_scores_boot)
                aucs.append(auc)
                
        aucs = np.array(aucs)
        
        # 计算置信区间
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
        """
        置换检验比较两个方法
        
        Args:
            y_true: 真实标签
            y_scores_a: 方法A的预测分数
            y_scores_b: 方法B的预测分数
            n_permutations: 置换次数
            
        Returns:
            observed_diff: 观察到的差异
            p_value: p值
        """
        # 观察到的AUC差异
        auc_a = roc_auc_score(y_true, y_scores_a)
        auc_b = roc_auc_score(y_true, y_scores_b)
        observed_diff = auc_a - auc_b
        
        # 置换检验
        count = 0
        for _ in range(n_permutations):
            # 随机交换两个方法的预测
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
    # 测试代码
    print("Testing Evaluation Metrics Module...")
    
    np.random.seed(42)
    
    # 创建测试数据
    n_samples = 1000
    y_true = np.random.randint(0, 2, n_samples)
    
    # 模拟不同质量的预测
    # 好的预测器
    y_scores_good = y_true * 0.7 + np.random.rand(n_samples) * 0.3
    y_scores_good = np.clip(y_scores_good, 0, 1)
    
    # 差的预测器
    y_scores_bad = np.random.rand(n_samples)
    
    # 测试评估器
    print("\nTesting MembershipInferenceEvaluator...")
    evaluator = MembershipInferenceEvaluator()
    
    metrics_good = evaluator.evaluate(y_true, y_scores_good)
    print("\nGood predictor metrics:")
    print(metrics_good)
    
    metrics_bad = evaluator.evaluate(y_true, y_scores_bad)
    print("\nBad predictor metrics:")
    print(metrics_bad)
    
    # 测试最优阈值
    threshold, score = evaluator.find_optimal_threshold(y_true, y_scores_good, "f1")
    print(f"\nOptimal threshold: {threshold:.3f}, F1: {score:.4f}")
    
    # 测试基线对比
    print("\nTesting BaselineComparator...")
    comparator = BaselineComparator()
    comparator.add_result("AMSM", y_true, y_scores_good)
    comparator.add_result("Random", y_true, y_scores_bad)
    
    comparator.print_comparison_table()
    
    improvements = comparator.compute_improvement("AMSM", "Random")
    print(f"\nImprovements over Random: {improvements}")
    
    # 测试多粒度评估
    print("\nTesting MultiGranularityEvaluator...")
    multi_eval = MultiGranularityEvaluator()
    multi_eval.evaluate_granularity("document", y_true, y_scores_good)
    multi_eval.evaluate_granularity("paragraph", y_true, y_scores_good * 0.9)
    multi_eval.evaluate_granularity("sentence", y_true, y_scores_good * 0.8)
    
    multi_eval.print_granularity_comparison()
    
    # 测试统计显著性
    print("\nTesting StatisticalTester...")
    mean_auc, lower, upper = StatisticalTester.bootstrap_auc(y_true, y_scores_good)
    print(f"Bootstrap AUC: {mean_auc:.4f} ({lower:.4f}, {upper:.4f})")
    
    diff, p_value = StatisticalTester.permutation_test(y_true, y_scores_good, y_scores_bad)
    print(f"Permutation test: diff={diff:.4f}, p-value={p_value:.4f}")
    
    print("\nAll tests passed!")
