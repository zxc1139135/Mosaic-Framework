"""
Training Module
===============

四阶段训练模块，实现AMSM框架的完整训练流程。

Stages:
    - Phase I: Expert System Construction (专家体系构建)
    - Phase II: Domain Specialization Training (领域专业化训练)
    - Phase III: Ensemble Mechanism Training (集成机制训练)
    - Phase IV: Attack Execution (攻击执行)
"""

from .expert_trainer import (
    ExpertTrainingConfig,
    DistillationLoss,
    DiversityLoss,
    CrossExpertDistillationLoss,
    ExpertTrainer,
    MultiExpertTrainer,
)

from .ensemble_trainer import (
    EnsembleTrainingConfig,
    RouterLoss,
    MetaLearnerLoss,
    EnsembleTrainer,
)

from .attack_trainer import (
    AttackTrainingConfig,
    EarlyStopping,
    AttackTrainer,
    FeatureCollector,
    AttackPipeline,
)

__all__ = [
    # Expert Training
    "ExpertTrainingConfig",
    "DistillationLoss",
    "DiversityLoss",
    "CrossExpertDistillationLoss",
    "ExpertTrainer",
    "MultiExpertTrainer",
    
    # Ensemble Training
    "EnsembleTrainingConfig",
    "RouterLoss",
    "MetaLearnerLoss",
    "EnsembleTrainer",
    
    # Attack Training
    "AttackTrainingConfig",
    "EarlyStopping",
    "AttackTrainer",
    "FeatureCollector",
    "AttackPipeline",
]
