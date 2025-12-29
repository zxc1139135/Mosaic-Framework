"""
AMSM Framework - Unit Tests
===========================

测试框架核心组件的功能正确性。

运行测试:
    pytest tests/test_models.py -v
    pytest tests/ -v --cov=src
"""

import sys
import os
from pathlib import Path

# 添加项目根目录
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import numpy as np
import torch
import torch.nn as nn


# ==================== 模型测试 ====================

class TestExpertModel:
    """测试专家模型"""
    
    def test_expert_model_small_creation(self):
        """测试小型专家模型创建"""
        from src.models import ExpertModelSmall
        
        model = ExpertModelSmall(
            expert_id=0,
            vocab_size=1000,
            hidden_dim=256,
            num_layers=2,
        )
        
        assert model is not None
        assert model.expert_id == 0
        
    def test_expert_model_forward(self):
        """测试专家模型前向传播"""
        from src.models import ExpertModelSmall
        
        model = ExpertModelSmall(
            expert_id=0,
            vocab_size=1000,
            hidden_dim=256,
            num_layers=2,
        )
        
        batch_size = 4
        seq_length = 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        output = model(input_ids)
        
        assert output.shape == (batch_size, seq_length, 1000)
        
    def test_multi_expert_system(self):
        """测试多专家系统"""
        from src.models import MultiExpertSystem
        
        system = MultiExpertSystem(
            num_experts=4,
            vocab_size=1000,
            hidden_dim=256,
            num_layers=2,
            use_small_model=True,
        )
        
        assert len(system.experts) == 4
        assert system.get_total_params() > 0
        
    def test_multi_expert_forward_all(self):
        """测试所有专家前向传播"""
        from src.models import MultiExpertSystem
        
        system = MultiExpertSystem(
            num_experts=4,
            vocab_size=1000,
            hidden_dim=256,
            num_layers=2,
            use_small_model=True,
        )
        
        batch_size = 2
        seq_length = 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        outputs = system.forward_all_experts(input_ids)
        
        assert len(outputs) == 4
        for output in outputs:
            assert output.shape == (batch_size, seq_length, 1000)


class TestRouterNetwork:
    """测试路由网络"""
    
    def test_router_creation(self):
        """测试路由网络创建"""
        from src.models import RouterNetwork
        
        router = RouterNetwork(
            input_dim=768,
            num_experts=8,
            hidden_dims=[256],
            top_k=3,
        )
        
        assert router is not None
        
    def test_router_forward(self):
        """测试路由网络前向传播"""
        from src.models import RouterNetwork
        
        router = RouterNetwork(
            input_dim=768,
            num_experts=8,
            hidden_dims=[256],
            top_k=3,
        )
        
        batch_size = 4
        features = torch.randn(batch_size, 768)
        
        weights, top_k_indices, top_k_weights = router(features)
        
        assert weights.shape == (batch_size, 8)
        assert top_k_indices.shape == (batch_size, 3)
        assert top_k_weights.shape == (batch_size, 3)
        
        # 权重应该归一化
        assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)
        
    def test_simple_feature_extractor(self):
        """测试简化特征提取器"""
        from src.models import SimpleFeatureExtractor
        
        extractor = SimpleFeatureExtractor(
            vocab_size=1000,
            output_dim=768,
        )
        
        batch_size = 4
        seq_length = 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_length))
        
        features = extractor(input_ids)
        
        assert features.shape == (batch_size, 768)


class TestMetaLearner:
    """测试元学习器"""
    
    def test_meta_learner_creation(self):
        """测试元学习器创建"""
        from src.models import MetaLearner
        
        learner = MetaLearner(
            input_dim=256,
            hidden_dims=[128, 64],
            dropout=0.2,
        )
        
        assert learner is not None
        
    def test_meta_learner_forward(self):
        """测试元学习器前向传播"""
        from src.models import MetaLearner
        
        learner = MetaLearner(
            input_dim=256,
            hidden_dims=[128, 64],
            dropout=0.2,
        )
        
        batch_size = 4
        features = torch.randn(batch_size, 256)
        
        output = learner(features)
        
        assert output.shape == (batch_size, 1)
        
    def test_meta_learner_predict(self):
        """测试元学习器预测"""
        from src.models import MetaLearner
        
        learner = MetaLearner(
            input_dim=256,
            hidden_dims=[128, 64],
        )
        learner.eval()
        
        batch_size = 4
        features = torch.randn(batch_size, 256)
        
        probs = learner.predict_proba(features)
        preds = learner.predict(features)
        
        assert probs.shape == (batch_size, 1)
        assert preds.shape == (batch_size, 1)
        assert (probs >= 0).all() and (probs <= 1).all()
        assert ((preds == 0) | (preds == 1)).all()


class TestAttackClassifier:
    """测试攻击分类器"""
    
    def test_attack_classifier_creation(self):
        """测试攻击分类器创建"""
        from src.models import AttackClassifier
        
        classifier = AttackClassifier(
            input_dim=45,
            hidden_dims=[128, 64],
            dropout=0.3,
        )
        
        assert classifier is not None
        
    def test_attack_classifier_forward(self):
        """测试攻击分类器前向传播"""
        from src.models import AttackClassifier
        
        classifier = AttackClassifier(
            input_dim=45,
            hidden_dims=[128, 64],
        )
        
        batch_size = 8
        features = torch.randn(batch_size, 45)
        
        output = classifier(features)
        
        assert output.shape == (batch_size, 1)


# ==================== 数据测试 ====================

class TestDataLoader:
    """测试数据加载"""
    
    def test_text_dataset(self):
        """测试文本数据集"""
        from src.data import TextDataset
        
        texts = ["Hello world", "Test text", "Another sample"]
        dataset = TextDataset(texts, max_length=32)
        
        assert len(dataset) == 3
        
        sample = dataset[0]
        assert "input_ids" in sample
        assert sample["input_ids"].shape[0] <= 32
        
    def test_membership_dataset(self):
        """测试成员数据集"""
        from src.data import MembershipDataset
        
        member_texts = ["Member text 1", "Member text 2"]
        non_member_texts = ["Non-member text 1", "Non-member text 2"]
        
        dataset = MembershipDataset(
            member_texts=member_texts,
            non_member_texts=non_member_texts,
            max_length=32,
        )
        
        assert len(dataset) == 4
        assert dataset.get_member_ratio() == 0.5
        
    def test_data_loader_factory(self):
        """测试数据加载器工厂"""
        from src.data import DataConfig, DataLoaderFactory
        
        config = DataConfig(batch_size=2, max_length=32)
        factory = DataLoaderFactory(config)
        
        texts = ["Text 1", "Text 2", "Text 3", "Text 4"]
        dataloader = factory.create_text_dataloader(texts)
        
        batch = next(iter(dataloader))
        assert "input_ids" in batch
        assert batch["input_ids"].shape[0] == 2


class TestDomainClustering:
    """测试领域聚类"""
    
    def test_complexity_analyzer(self):
        """测试复杂度分析器"""
        from src.data import DomainComplexityAnalyzer
        
        analyzer = DomainComplexityAnalyzer()
        
        texts = [
            "This is a simple sentence.",
            "Another simple text here.",
        ]
        
        complexity = analyzer.compute_complexity(texts)
        
        assert 0 <= complexity <= 1
        
    def test_domain_clusterer(self):
        """测试领域聚类器"""
        from src.data import DomainClusterer, SimpleEmbeddingExtractor
        
        extractor = SimpleEmbeddingExtractor(max_features=100)
        clusterer = DomainClusterer(num_clusters=2)
        
        texts = [
            "Machine learning is great",
            "Deep learning models",
            "Cooking recipes are fun",
            "Kitchen appliances review",
        ]
        
        embeddings = extractor.extract_embeddings(texts)
        labels = clusterer.fit(embeddings)
        
        assert len(labels) == 4
        assert set(labels).issubset({0, 1})


# ==================== 训练测试 ====================

class TestTraining:
    """测试训练模块"""
    
    def test_distillation_loss(self):
        """测试蒸馏损失"""
        from src.training import DistillationLoss
        
        loss_fn = DistillationLoss(temperature=2.0)
        
        batch_size = 4
        vocab_size = 100
        
        student_logits = torch.randn(batch_size, vocab_size)
        teacher_logits = torch.randn(batch_size, vocab_size)
        
        loss = loss_fn(student_logits, teacher_logits)
        
        assert loss.dim() == 0  # 标量
        assert loss.item() >= 0
        
    def test_router_loss(self):
        """测试路由损失"""
        from src.training import RouterLoss
        
        loss_fn = RouterLoss(num_experts=8)
        
        batch_size = 4
        task_loss = torch.tensor(1.0)
        routing_weights = torch.softmax(torch.randn(batch_size, 8), dim=-1)
        
        total_loss, loss_dict = loss_fn(task_loss, routing_weights)
        
        assert "task_loss" in loss_dict
        assert "balance_loss" in loss_dict
        
    def test_early_stopping(self):
        """测试早停机制"""
        from src.training import EarlyStopping
        
        early_stopping = EarlyStopping(patience=3)
        
        # 模拟损失下降
        assert not early_stopping(1.0)
        assert not early_stopping(0.9)
        assert not early_stopping(0.8)
        
        # 模拟损失不再下降
        assert not early_stopping(0.85)
        assert not early_stopping(0.85)
        assert early_stopping(0.85)  # 第3次，触发早停


# ==================== 评估测试 ====================

class TestEvaluation:
    """测试评估模块"""
    
    def test_evaluator_creation(self):
        """测试评估器创建"""
        from src.evaluation import MembershipInferenceEvaluator
        
        evaluator = MembershipInferenceEvaluator()
        assert evaluator is not None
        
    def test_auc_computation(self):
        """测试AUC计算"""
        from src.evaluation import MembershipInferenceEvaluator
        
        evaluator = MembershipInferenceEvaluator()
        
        # 完美预测
        y_true = np.array([0, 0, 1, 1])
        y_scores = np.array([0.1, 0.2, 0.8, 0.9])
        
        auc = evaluator.compute_auc_roc(y_true, y_scores)
        
        assert auc == 1.0
        
    def test_full_evaluation(self):
        """测试完整评估"""
        from src.evaluation import MembershipInferenceEvaluator
        
        evaluator = MembershipInferenceEvaluator()
        
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_scores = y_true * 0.7 + np.random.rand(100) * 0.3
        
        metrics = evaluator.evaluate(y_true, y_scores)
        
        assert hasattr(metrics, "auc_roc")
        assert hasattr(metrics, "accuracy")
        assert hasattr(metrics, "attack_advantage")
        assert 0 <= metrics.auc_roc <= 1
        
    def test_baseline_comparator(self):
        """测试基线对比器"""
        from src.evaluation import BaselineComparator
        
        comparator = BaselineComparator()
        
        y_true = np.array([0, 0, 1, 1, 0, 1])
        scores_a = np.array([0.1, 0.2, 0.8, 0.9, 0.3, 0.7])
        scores_b = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        
        comparator.add_result("Method A", y_true, scores_a)
        comparator.add_result("Method B", y_true, scores_b)
        
        comparison = comparator.compare()
        
        assert "Method A" in comparison
        assert "Method B" in comparison
        assert comparison["Method A"]["auc_roc"] > comparison["Method B"]["auc_roc"]


# ==================== 工具测试 ====================

class TestUtils:
    """测试工具函数"""
    
    def test_set_seed(self):
        """测试随机种子设置"""
        from src.utils import set_seed
        
        set_seed(42)
        a = torch.rand(5)
        
        set_seed(42)
        b = torch.rand(5)
        
        assert torch.equal(a, b)
        
    def test_average_meter(self):
        """测试平均值计算器"""
        from src.utils import AverageMeter
        
        meter = AverageMeter("test")
        
        meter.update(1.0)
        meter.update(2.0)
        meter.update(3.0)
        
        assert meter.avg == 2.0
        assert meter.count == 3
        
    def test_timer(self):
        """测试计时器"""
        from src.utils import Timer
        import time
        
        timer = Timer()
        timer.start()
        time.sleep(0.1)
        elapsed = timer.stop()
        
        assert elapsed >= 0.1
        
    def test_count_parameters(self):
        """测试参数计数"""
        from src.utils import count_parameters
        
        model = nn.Linear(10, 5)
        params = count_parameters(model)
        
        # 10*5 weights + 5 bias = 55
        assert params == 55


# ==================== 集成测试 ====================

class TestIntegration:
    """集成测试"""
    
    def test_full_pipeline_small(self):
        """测试小规模完整流程"""
        from src.models import ExpertModelSmall, RouterNetwork, MetaLearner, AttackClassifier
        from src.data import DataConfig, DataLoaderFactory
        from src.evaluation import MembershipInferenceEvaluator
        
        # 创建模型
        expert = ExpertModelSmall(expert_id=0, vocab_size=100, hidden_dim=64, num_layers=1)
        router = RouterNetwork(input_dim=64, num_experts=2, hidden_dims=[32], top_k=1)
        meta_learner = MetaLearner(input_dim=32, hidden_dims=[16])
        classifier = AttackClassifier(input_dim=10, hidden_dims=[8])
        
        # 创建数据
        texts = ["Sample text " + str(i) for i in range(10)]
        config = DataConfig(batch_size=2, max_length=16)
        factory = DataLoaderFactory(config)
        dataloader = factory.create_text_dataloader(texts)
        
        # 前向传播测试
        batch = next(iter(dataloader))
        input_ids = batch["input_ids"]
        
        expert_output = expert(input_ids)
        assert expert_output.shape[0] == 2
        
        # 评估测试
        evaluator = MembershipInferenceEvaluator()
        y_true = np.array([0, 1, 0, 1])
        y_scores = np.array([0.2, 0.8, 0.3, 0.7])
        metrics = evaluator.evaluate(y_true, y_scores)
        
        assert metrics.auc_roc > 0.5


# ==================== 运行测试 ====================

if __name__ == "__main__":
    # 运行所有测试
    pytest.main([__file__, "-v", "--tb=short"])
