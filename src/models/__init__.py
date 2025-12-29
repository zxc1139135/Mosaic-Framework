"""
Models Module
=============

神经网络模型定义模块，包含AMSM框架的所有核心网络组件。

Components:
    - ExpertModel: 专家模型 (Decoder-only Transformer)
    - RouterNetwork: 动态路由选择网络
    - MetaLearner: 元学习决策器
    - AttackClassifier: 成员推理攻击分类器
"""

from .expert_model import (
    ExpertModel,
    ExpertModelSmall,
    MultiExpertSystem,
    TransformerBlock,
    MultiHeadAttention,
    FeedForward,
    RMSNorm,
)

from .router_network import (
    RouterNetwork,
    FeatureExtractor,
    SimpleFeatureExtractor,
    GatingNetwork,
    AttentionRouter,
    HierarchicalRouter,
)

from .meta_learner import (
    MetaLearner,
    MetaFeatureBuilder,
    EnsembleDecisionMaker,
    WeightedEnsemble,
    AttentionFusion,
)

from .attack_classifier import (
    AttackClassifier,
    AttackFeatureExtractor,
    MembershipInferenceAttack,
    BaselineAttacks,
)

__all__ = [
    # Expert Model
    "ExpertModel",
    "ExpertModelSmall",
    "MultiExpertSystem",
    "TransformerBlock",
    "MultiHeadAttention",
    "FeedForward",
    "RMSNorm",
    
    # Router Network
    "RouterNetwork",
    "FeatureExtractor",
    "SimpleFeatureExtractor",
    "GatingNetwork",
    "AttentionRouter",
    "HierarchicalRouter",
    
    # Meta Learner
    "MetaLearner",
    "MetaFeatureBuilder",
    "EnsembleDecisionMaker",
    "WeightedEnsemble",
    "AttentionFusion",
    
    # Attack Classifier
    "AttackClassifier",
    "AttackFeatureExtractor",
    "MembershipInferenceAttack",
    "BaselineAttacks",
]
