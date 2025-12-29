"""
Models Module
=============

Components:
    - ExpertModel
    - RouterNetwork
    - MetaLearner
    - AttackClassifier
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
