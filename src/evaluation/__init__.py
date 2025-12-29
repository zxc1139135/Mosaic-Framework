"""
Evaluation Module
=================

攻击评估和性能分析模块。

Components:
    - MembershipInferenceEvaluator: 成员推理攻击评估器
    - BaselineComparator: 基线方法对比器
    - MultiGranularityEvaluator: 多粒度评估器
    - StatisticalTester: 统计显著性测试
"""

from .metrics import (
    AttackMetrics,
    MembershipInferenceEvaluator,
    BaselineComparator,
    MultiGranularityEvaluator,
    StatisticalTester,
)

__all__ = [
    "AttackMetrics",
    "MembershipInferenceEvaluator",
    "BaselineComparator",
    "MultiGranularityEvaluator",
    "StatisticalTester",
]
