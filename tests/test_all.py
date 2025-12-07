#!/usr/bin/env python
"""
Mosaic Framework - Comprehensive Tests
=====================================

。

Usage:
 python tests/test_all.py
 python tests/test_all.py --module models
"""

import os
import sys
import argparse
import unittest
from pathlib import Path

# 
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn

class TestExpertModels(unittest.TestCase):
 """"""
 
 def setUp(self):
 """"""
 self.batch_size = 4
 self.seq_length = 64
 self.vocab_size = 1000
 self.hidden_dim = 256
 
 def test_expert_model_small_forward(self):
 """"""
 from src.models import ExpertModelSmall
 
 model = ExpertModelSmall(
 expert_id=0,
 vocab_size=self.vocab_size,
 hidden_dim=self.hidden_dim,
 num_layers=2,
 )
 
 input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
 output = model(input_ids)
 
 self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.vocab_size))
 
 def test_expert_model_small_loss(self):
 """"""
 from src.models import ExpertModelSmall
 
 model = ExpertModelSmall(
 expert_id=0,
 vocab_size=self.vocab_size,
 hidden_dim=self.hidden_dim,
 num_layers=2,
 )
 
 input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
 labels = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
 
 loss = model.compute_loss(input_ids, labels)
 
 self.assertIsInstance(loss.item(), float)
 self.assertGreater(loss.item(), 0)
 
 def test_multi_expert_system(self):
 """"""
 from src.models import MultiExpertSystem
 
 num_experts = 4
 system = MultiExpertSystem(
 num_experts=num_experts,
 vocab_size=self.vocab_size,
 hidden_dim=self.hidden_dim,
 num_layers=2,
 use_small_model=True,
 )
 
 input_ids = torch.randint(0, self.vocab_size, (self.batch_size, self.seq_length))
 
 # 
 output = system.forward(input_ids, expert_id=0)
 self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.vocab_size))
 
 # 
 all_outputs = system.forward_all_experts(input_ids)
 self.assertEqual(len(all_outputs), num_experts)

class TestRouterNetwork(unittest.TestCase):
 """"""
 
 def setUp(self):
 """"""
 self.batch_size = 8
 self.input_dim = 768
 self.num_experts = 8
 
 def test_router_forward(self):
 """"""
 from src.models import RouterNetwork
 
 router = RouterNetwork(
 input_dim=self.input_dim,
 num_experts=self.num_experts,
 hidden_dims=[256],
 top_k=3,
 )
 
 features = torch.randn(self.batch_size, self.input_dim)
 weights, top_k_indices, top_k_weights = router(features)
 
 self.assertEqual(weights.shape, (self.batch_size, self.num_experts))
 self.assertEqual(top_k_indices.shape, (self.batch_size, 3))
 self.assertEqual(top_k_weights.shape, (self.batch_size, 3))
 
 # 1
 weight_sums = weights.sum(dim=-1)
 self.assertTrue(torch.allclose(weight_sums, torch.ones(self.batch_size), atol=1e-5))
 
 def test_router_balance_loss(self):
 """"""
 from src.models import RouterNetwork
 
 router = RouterNetwork(
 input_dim=self.input_dim,
 num_experts=self.num_experts,
 )
 
 features = torch.randn(self.batch_size, self.input_dim)
 weights, _, _ = router(features)
 
 balance_loss = router.compute_balance_loss(weights)
 self.assertIsInstance(balance_loss.item(), float)

class TestMetaLearner(unittest.TestCase):
 """"""
 
 def setUp(self):
 """"""
 self.batch_size = 16
 self.input_dim = 128
 
 def test_meta_learner_forward(self):
 """"""
 from src.models import MetaLearner
 
 meta_learner = MetaLearner(
 input_dim=self.input_dim,
 hidden_dims=[64, 32],
 )
 
 features = torch.randn(self.batch_size, self.input_dim)
 output = meta_learner(features)
 
 self.assertEqual(output.shape, (self.batch_size, 1))
 
 def test_meta_learner_predict(self):
 """"""
 from src.models import MetaLearner
 
 meta_learner = MetaLearner(
 input_dim=self.input_dim,
 hidden_dims=[64, 32],
 )
 
 features = torch.randn(self.batch_size, self.input_dim)
 
 probs = meta_learner.predict_proba(features)
 self.assertTrue((probs >= 0).all() and (probs <= 1).all())
 
 preds = meta_learner.predict(features)
 self.assertTrue(((preds == 0) | (preds == 1)).all())

class TestAttackClassifier(unittest.TestCase):
 """"""
 
 def setUp(self):
 """"""
 self.batch_size = 32
 self.feature_dim = 45
 
 def test_attack_classifier_forward(self):
 """"""
 from src.models import AttackClassifier
 
 classifier = AttackClassifier(
 input_dim=self.feature_dim,
 hidden_dims=[128, 64],
 )
 
 features = torch.randn(self.batch_size, self.feature_dim)
 output = classifier(features)
 
 self.assertEqual(output.shape, (self.batch_size, 1))
 
 def test_attack_feature_extractor(self):
 """"""
 from src.models import AttackFeatureExtractor
 
 extractor = AttackFeatureExtractor(num_experts=8)
 
 batch_size = 4
 seq_length = 64
 vocab_size = 1000
 
 target_probs = torch.softmax(torch.randn(batch_size, seq_length, vocab_size), dim=-1)
 target_loss = torch.rand(batch_size)
 input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
 
 expert_probs = [
 torch.softmax(torch.randn(batch_size, seq_length, vocab_size), dim=-1)
 for _ in range(8)
 ]
 routing_weights = torch.softmax(torch.randn(batch_size, 8), dim=-1)
 
 features = extractor(target_probs, target_loss, input_ids, expert_probs, routing_weights)
 
 self.assertEqual(features.shape[0], batch_size)
 self.assertEqual(features.shape[1], 45) # 12 + 28 + 5

class TestDataLoading(unittest.TestCase):
 """"""
 
 def test_text_dataset(self):
 """"""
 from src.data import TextDataset
 from transformers import AutoTokenizer
 
 texts = ["Hello world", "This is a test", "Machine learning"]
 
 # tokenizer
 class SimpleTokenizer:
 def __call__(self, text, max_length=512, truncation=True, padding="max_length", return_tensors="pt"):
 # tokenization
 ids = [ord(c) % 1000 for c in text[:max_length]]
 ids = ids + [0] * (max_length - len(ids))
 return {
 "input_ids": torch.tensor([ids]),
 "attention_mask": torch.tensor([[1] * len(text[:max_length]) + [0] * (max_length - len(text[:max_length]))])
 }
 
 tokenizer = SimpleTokenizer()
 dataset = TextDataset(texts, tokenizer, max_length=32)
 
 self.assertEqual(len(dataset), 3)
 
 item = dataset[0]
 self.assertIn("input_ids", item)
 
 def test_membership_dataset(self):
 """"""
 from src.data import MembershipDataset
 
 member_texts = ["member 1", "member 2"]
 non_member_texts = ["non-member 1", "non-member 2"]
 
 class SimpleTokenizer:
 def __call__(self, text, max_length=512, truncation=True, padding="max_length", return_tensors="pt"):
 ids = [ord(c) % 1000 for c in text[:max_length]]
 ids = ids + [0] * (max_length - len(ids))
 return {
 "input_ids": torch.tensor([ids]),
 "attention_mask": torch.tensor([[1] * min(len(text), max_length) + [0] * max(0, max_length - len(text))])
 }
 
 tokenizer = SimpleTokenizer()
 dataset = MembershipDataset(member_texts, non_member_texts, tokenizer, max_length=32)
 
 self.assertEqual(len(dataset), 4)
 
 # 
 labels = [dataset[i]["label"] for i in range(len(dataset))]
 self.assertEqual(sum(labels), 2) # 2

class TestDomainClustering(unittest.TestCase):
 """"""
 
 def test_complexity_analyzer(self):
 """"""
 from src.data import DomainComplexityAnalyzer
 
 analyzer = DomainComplexityAnalyzer()
 
 text = "This is a simple test sentence. It has some words and punctuation."
 complexity = analyzer.compute_complexity(text)
 
 self.assertIsInstance(complexity, float)
 self.assertGreaterEqual(complexity, 0)
 self.assertLessEqual(complexity, 1)
 
 def test_domain_clusterer(self):
 """"""
 from src.data import DomainClusterer
 
 # 
 embeddings = np.random.randn(100, 128)
 
 clusterer = DomainClusterer(num_clusters=4)
 labels = clusterer.fit(embeddings)
 
 self.assertEqual(len(labels), 100)
 self.assertEqual(len(set(labels)), 4)

class TestTraining(unittest.TestCase):
 """"""
 
 def test_distillation_loss(self):
 """"""
 from src.training import DistillationLoss
 
 loss_fn = DistillationLoss(temperature=2.0)
 
 batch_size = 4
 seq_length = 32
 vocab_size = 100
 
 student_logits = torch.randn(batch_size, seq_length, vocab_size)
 teacher_logits = torch.randn(batch_size, seq_length, vocab_size)
 
 loss = loss_fn(student_logits, teacher_logits)
 
 self.assertIsInstance(loss.item(), float)
 self.assertGreater(loss.item(), 0)
 
 def test_early_stopping(self):
 """"""
 from src.training import EarlyStopping
 
 early_stopping = EarlyStopping(patience=3)
 
 # 
 self.assertFalse(early_stopping(1.0))
 self.assertFalse(early_stopping(0.9))
 self.assertFalse(early_stopping(0.8))
 
 # 
 self.assertFalse(early_stopping(0.85))
 self.assertFalse(early_stopping(0.85))
 self.assertTrue(early_stopping(0.85)) # 3

class TestEvaluation(unittest.TestCase):
 """"""
 
 def setUp(self):
 """"""
 np.random.seed(42)
 self.n_samples = 100
 self.y_true = np.random.randint(0, 2, self.n_samples)
 self.y_scores = self.y_true * 0.7 + np.random.rand(self.n_samples) * 0.3
 
 def test_evaluator_metrics(self):
 """"""
 from src.evaluation import MembershipInferenceEvaluator
 
 evaluator = MembershipInferenceEvaluator()
 metrics = evaluator.evaluate(self.y_true, self.y_scores)
 
 self.assertGreater(metrics.auc_roc, 0.5) # 
 self.assertGreater(metrics.accuracy, 0.5)
 self.assertGreater(metrics.attack_advantage, 0)
 
 def test_tpr_at_fpr(self):
 """TPR@FPR"""
 from src.evaluation import MembershipInferenceEvaluator
 
 evaluator = MembershipInferenceEvaluator(fpr_thresholds=[0.01, 0.1])
 tpr_dict = evaluator.compute_tpr_at_fpr(self.y_true, self.y_scores)
 
 self.assertIn(0.01, tpr_dict)
 self.assertIn(0.1, tpr_dict)
 
 def test_baseline_comparator(self):
 """"""
 from src.evaluation import BaselineComparator
 
 comparator = BaselineComparator()
 comparator.add_result("Method_A", self.y_true, self.y_scores)
 comparator.add_result("Method_B", self.y_true, np.random.rand(self.n_samples))
 
 comparison = comparator.compare()
 
 self.assertIn("Method_A", comparison)
 self.assertIn("Method_B", comparison)
 
 def test_statistical_tester(self):
 """"""
 from src.evaluation import StatisticalTester
 
 mean_auc, lower, upper = StatisticalTester.bootstrap_auc(
 self.y_true, self.y_scores, n_bootstrap=100
 )
 
 self.assertLessEqual(lower, mean_auc)
 self.assertGreaterEqual(upper, mean_auc)

class TestUtils(unittest.TestCase):
 """"""
 
 def test_set_seed(self):
 """"""
 from src.utils import set_seed
 
 set_seed(42)
 a = torch.rand(10)
 
 set_seed(42)
 b = torch.rand(10)
 
 self.assertTrue(torch.equal(a, b))
 
 def test_average_meter(self):
 """"""
 from src.utils import AverageMeter
 
 meter = AverageMeter("test")
 
 meter.update(1.0)
 meter.update(2.0)
 meter.update(3.0)
 
 self.assertEqual(meter.avg, 2.0)
 self.assertEqual(meter.count, 3)
 
 def test_timer(self):
 """"""
 from src.utils import Timer
 import time
 
 with Timer() as timer:
 time.sleep(0.1)
 
 self.assertGreater(timer.elapsed, 0.09)
 self.assertLess(timer.elapsed, 0.2)

def run_tests(module: str = None):
 """"""
 loader = unittest.TestLoader()
 suite = unittest.TestSuite()
 
 test_classes = {
 "models": [TestExpertModels, TestRouterNetwork, TestMetaLearner, TestAttackClassifier],
 "data": [TestDataLoading, TestDomainClustering],
 "training": [TestTraining],
 "evaluation": [TestEvaluation],
 "utils": [TestUtils],
 }
 
 if module and module in test_classes:
 for test_class in test_classes[module]:
 suite.addTests(loader.loadTestsFromTestCase(test_class))
 else:
 for test_class_list in test_classes.values():
 for test_class in test_class_list:
 suite.addTests(loader.loadTestsFromTestCase(test_class))
 
 runner = unittest.TextTestRunner(verbosity=2)
 result = runner.run(suite)
 
 return result.wasSuccessful()

if __name__ == "__main__":
 parser = argparse.ArgumentParser(description="Mosaic Framework Tests")
 parser.add_argument(
 "--module",
 type=str,
 choices=["models", "data", "training", "evaluation", "utils"],
 default=None,
 help="Specific module to test",
 )
 
 args = parser.parse_args()
 
 success = run_tests(args.module)
 sys.exit(0 if success else 1)
