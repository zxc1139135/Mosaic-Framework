#!/usr/bin/env python
"""
Mosaic Framework - Quick Start Example
====================================

，。

Usage:
 python scripts/quickstart.py
"""

import os
import sys
from pathlib import Path

# 
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch

# 
torch.manual_seed(42)
np.random.seed(42)

print("=" * 60)
print("Mosaic Framework Quick Start Example")
print("=" * 60)

# ==================== 1. ====================
print("\n[Step 1] Creating Expert Models...")

from src.models import ExpertModelSmall, MultiExpertSystem

# 
expert = ExpertModelSmall(
 expert_id=0,
 vocab_size=1000,
 hidden_dim=256,
 num_layers=2,
 num_heads=4,
)
print(f" Single expert parameters: {sum(p.numel() for p in expert.parameters()):,}")

# 
num_experts = 4
multi_expert = MultiExpertSystem(
 num_experts=num_experts,
 vocab_size=1000,
 hidden_dim=256,
 num_layers=2,
 use_small_model=True,
)
print(f" Multi-expert system with {num_experts} experts")
print(f" Total parameters: {multi_expert.get_total_params():,}")

# 
batch_size = 2
seq_length = 32
input_ids = torch.randint(0, 1000, (batch_size, seq_length))

output = expert(input_ids)
print(f" Expert output shape: {output.shape}")

# ==================== 2. ====================
print("\n[Step 2] Creating Router Network...")

from src.models import RouterNetwork, SimpleFeatureExtractor

# 
feature_extractor = SimpleFeatureExtractor(
 vocab_size=1000,
 output_dim=256,
)

# 
router = RouterNetwork(
 input_dim=256,
 num_experts=num_experts,
 hidden_dims=[128],
 top_k=2,
)

# 
features = feature_extractor(input_ids)
weights, top_k_indices, top_k_weights = router(features)
print(f" Feature shape: {features.shape}")
print(f" Routing weights shape: {weights.shape}")
print(f" Top-K indices: {top_k_indices}")

# ==================== 3. ====================
print("\n[Step 3] Creating Meta Learner...")

from src.models import MetaLearner

meta_learner = MetaLearner(
 input_dim=64,
 hidden_dims=[32, 16],
)

# 
meta_features = torch.randn(batch_size, 64)
predictions = meta_learner.predict_proba(meta_features)
print(f" Meta features shape: {meta_features.shape}")
print(f" Predictions: {predictions.squeeze().tolist()}")

# ==================== 4. ====================
print("\n[Step 4] Creating Attack Classifier...")

from src.models import AttackClassifier, AttackFeatureExtractor

# 
attack_feature_extractor = AttackFeatureExtractor(num_experts=num_experts)

# 
attack_classifier = AttackClassifier(
 input_dim=45,
 hidden_dims=[64, 32],
)

print(f" Attack classifier parameters: {sum(p.numel() for p in attack_classifier.parameters()):,}")

# ==================== 5. ====================
print("\n[Step 5] Evaluation Metrics Example...")

from src.evaluation import MembershipInferenceEvaluator, BaselineComparator

# 
n_samples = 500
y_true = np.random.randint(0, 2, n_samples)
y_scores_good = y_true * 0.7 + np.random.rand(n_samples) * 0.3
y_scores_bad = np.random.rand(n_samples)

# 
evaluator = MembershipInferenceEvaluator()
metrics = evaluator.evaluate(y_true, y_scores_good)

print(f" AUC-ROC: {metrics.auc_roc:.4f}")
print(f" Accuracy: {metrics.accuracy:.4f}")
print(f" Attack Advantage: {metrics.attack_advantage:.4f}")
print(f" TPR@1%FPR: {metrics.tpr_at_fpr.get(0.01, 0):.4f}")

# 
print("\n Baseline Comparison:")
comparator = BaselineComparator()
comparator.add_result("Mosaic", y_true, y_scores_good)
comparator.add_result("Random", y_true, y_scores_bad)
comparator.print_comparison_table()

# ==================== 6. ====================
print("\n[Step 6] Domain Clustering Example...")

from src.data import DomainComplexityAnalyzer, DomainClusterer

# 
analyzer = DomainComplexityAnalyzer()
texts = [
 "This is a simple sentence.",
 "The complex theoretical framework involves multiple interconnected components.",
 "AI is cool.",
]

print(" Text Complexity Analysis:")
for text in texts:
 complexity = analyzer.compute_complexity(text)
 print(f" '{text[:30]}...' -> {complexity:.4f}")

# 
embeddings = np.random.randn(100, 64)
clusterer = DomainClusterer(num_clusters=4)
labels = clusterer.fit(embeddings)
print(f"\n Clustered 100 samples into 4 domains")
print(f" Cluster sizes: {[sum(labels == i) for i in range(4)]}")

# ==================== 7. ====================
print("\n[Step 7] Simple Training Loop Example...")

from src.training import AttackTrainingConfig, AttackTrainer

# 
features = np.random.randn(200, 45).astype(np.float32)
labels = np.random.randint(0, 2, 200).astype(np.float32)

# 
features[:, 0] = labels * 2 - 1 + np.random.randn(200) * 0.5

# 
config = AttackTrainingConfig(
 num_epochs=5,
 batch_size=32,
 learning_rate=1e-3,
 device="cpu",
)

# 
classifier = AttackClassifier(input_dim=45, hidden_dims=[32, 16])
trainer = AttackTrainer(classifier, config)

print(" Training attack classifier for 5 epochs...")
history = trainer.train(features, labels)

print(f" Final Test Accuracy: {history['test_accuracy']:.4f}")

# ==================== 8. ====================
print("\n[Step 8] Full Pipeline Overview...")
print("""
 Mosaic:
 
 1. Phase I: K*=8
 2. Phase II: 
 - 
 - 
 - 
 3. Phase III: 
 - 
 - 
 4. Phase IV: 
 - 45
 - 
 - 
 
 :
 $ python scripts/train.py --config configs/config.yaml
""")

print("=" * 60)
print("Quick Start Complete!")
print("=" * 60)
print("\nNext steps:")
print(" 1. Review the configuration: configs/config.yaml")
print(" 2. Run full training: python scripts/train.py")
print(" 3. Run experiments: python scripts/run_experiments.py")
print(" 4. Check the documentation: README.md")
