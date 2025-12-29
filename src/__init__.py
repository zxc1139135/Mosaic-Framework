"""
AMSM Framework - Adaptive Multi-Shadow Model Framework
=======================================================

A comprehensive framework for membership inference attacks on Large Language Models
using adaptive multi-shadow model approach.

Modules:
    - models: Neural network architectures (Expert, Router, MetaLearner, AttackClassifier)
    - data: Data loading and domain clustering utilities
    - training: Four-phase training pipeline
    - evaluation: Attack evaluation metrics and analysis
    - utils: Helper functions and utilities

Reference:
    基于自适应多影子模型的大语言模型成员推理攻击框架
"""

__version__ = "1.0.0"
__author__ = "AMSM Research Team"

from . import models
from . import data
from . import training
from . import evaluation
from . import utils
