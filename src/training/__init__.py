"""
Training Module
===============

，Mosaic。

Stages:
 - Phase I: Expert System Construction ()
 - Phase II: Domain Specialization Training ()
 - Phase III: Ensemble Mechanism Training ()
 - Phase IV: Attack Execution ()
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
