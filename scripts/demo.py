#!/usr/bin/env python
"""
AMSM Framework - Demo Script
============================

演示AMSM框架的基本使用方法。

Usage:
    python scripts/demo.py
"""

import sys
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

print("=" * 60)
print("AMSM Framework Demo")
print("=" * 60)

# 1. 测试模型创建
print("\n1. Creating Models...")

from src.models import (
    ExpertModelSmall,
    MultiExpertSystem,
    RouterNetwork,
    MetaLearner,
    AttackClassifier,
    SimpleFeatureExtractor,
)

# 专家模型
expert = ExpertModelSmall(
    expert_id=0,
    vocab_size=1000,
    hidden_dim=256,
    num_layers=2,
)
print(f"   Expert model params: {sum(p.numel() for p in expert.parameters()):,}")

# 多专家系统
multi_expert = MultiExpertSystem(
    num_experts=4,
    vocab_size=1000,
    hidden_dim=256,
    num_layers=2,
    use_small_model=True,
)
print(f"   Multi-expert system params: {multi_expert.get_total_params():,}")

# 路由网络
router = RouterNetwork(
    input_dim=768,
    num_experts=4,
    hidden_dims=[256],
    top_k=2,
)
print(f"   Router network params: {sum(p.numel() for p in router.parameters()):,}")

# 元学习器
meta_learner = MetaLearner(
    input_dim=128,
    hidden_dims=[64, 32],
)
print(f"   Meta learner params: {sum(p.numel() for p in meta_learner.parameters()):,}")

# 攻击分类器
attack_classifier = AttackClassifier(
    input_dim=45,
    hidden_dims=[64, 32],
)
print(f"   Attack classifier params: {sum(p.numel() for p in attack_classifier.parameters()):,}")

# 2. 测试数据处理
print("\n2. Testing Data Processing...")

from src.data import (
    DataConfig,
    TextDataset,
    MembershipDataset,
    DataLoaderFactory,
    DomainComplexityAnalyzer,
)

# 创建数据集
texts = [f"Sample text number {i} for testing." for i in range(20)]
dataset = TextDataset(texts, max_length=32)
print(f"   Text dataset size: {len(dataset)}")

# 成员数据集
member_texts = texts[:10]
non_member_texts = texts[10:]
membership_dataset = MembershipDataset(
    member_texts=member_texts,
    non_member_texts=non_member_texts,
    max_length=32,
)
print(f"   Membership dataset size: {len(membership_dataset)}")
print(f"   Member ratio: {membership_dataset.get_member_ratio():.2f}")

# 领域复杂度
analyzer = DomainComplexityAnalyzer()
complexity = analyzer.compute_complexity(texts)
print(f"   Domain complexity: {complexity:.4f}")

# 3. 测试模型前向传播
print("\n3. Testing Model Forward Pass...")

batch_size = 4
seq_length = 32
input_ids = torch.randint(0, 1000, (batch_size, seq_length))

# 专家模型前向
with torch.no_grad():
    expert_output = expert(input_ids)
print(f"   Expert output shape: {expert_output.shape}")

# 多专家系统前向
with torch.no_grad():
    all_outputs = multi_expert.forward_all_experts(input_ids)
print(f"   All experts outputs: {len(all_outputs)} tensors")

# 路由网络前向
features = torch.randn(batch_size, 768)
with torch.no_grad():
    weights, indices, top_k_weights = router(features)
print(f"   Router weights shape: {weights.shape}")
print(f"   Top-K indices shape: {indices.shape}")

# 元学习器前向
meta_features = torch.randn(batch_size, 128)
with torch.no_grad():
    meta_output = meta_learner(meta_features)
    meta_probs = meta_learner.predict_proba(meta_features)
print(f"   Meta learner output shape: {meta_output.shape}")
print(f"   Meta learner probs: {meta_probs.squeeze().numpy()}")

# 攻击分类器前向
attack_features = torch.randn(batch_size, 45)
with torch.no_grad():
    attack_output = attack_classifier(attack_features)
print(f"   Attack classifier output shape: {attack_output.shape}")

# 4. 测试评估
print("\n4. Testing Evaluation Metrics...")

from src.evaluation import (
    MembershipInferenceEvaluator,
    BaselineComparator,
)

# 生成测试数据
np.random.seed(42)
n_samples = 500
y_true = np.random.randint(0, 2, n_samples)
y_scores = y_true * 0.7 + np.random.rand(n_samples) * 0.3
y_scores = np.clip(y_scores, 0, 1)

# 评估
evaluator = MembershipInferenceEvaluator()
metrics = evaluator.evaluate(y_true, y_scores)

print(f"   AUC-ROC: {metrics.auc_roc:.4f}")
print(f"   AUPRC: {metrics.auprc:.4f}")
print(f"   Accuracy: {metrics.accuracy:.4f}")
print(f"   F1 Score: {metrics.f1:.4f}")
print(f"   Attack Advantage: {metrics.attack_advantage:.4f}")
print(f"   TPR@1%FPR: {metrics.tpr_at_fpr.get(0.01, 0):.4f}")

# 基线对比
comparator = BaselineComparator()
comparator.add_result("AMSM", y_true, y_scores)
comparator.add_result("Random", y_true, np.random.rand(n_samples))

print("\n   Baseline Comparison:")
comparator.print_comparison_table()

# 5. 测试工具函数
print("\n5. Testing Utility Functions...")

from src.utils import (
    set_seed,
    get_device,
    Timer,
    AverageMeter,
    count_parameters,
)

# 设备检测
device = get_device()
print(f"   Device: {device}")

# 参数计数
params = count_parameters(expert)
print(f"   Expert model parameters: {params:,}")

# 计时器
import time
with Timer() as timer:
    time.sleep(0.1)
print(f"   Timer elapsed: {timer}")

# 平均值计算
meter = AverageMeter("test")
for i in range(5):
    meter.update(i * 0.2)
print(f"   Average meter: {meter}")

# 6. 完成
print("\n" + "=" * 60)
print("Demo completed successfully!")
print("=" * 60)
print("\nNext steps:")
print("  1. Run full training: python scripts/train.py")
print("  2. Run experiments: python scripts/run_experiments.py")
print("  3. Run tests: pytest tests/ -v")
