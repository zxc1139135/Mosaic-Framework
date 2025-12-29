"""
A comprehensive framework for membership inference attacks on Large Language Models
using adaptive multi-shadow model approach.

Modules:
    - models: Neural network architectures (Expert, Router, MetaLearner, AttackClassifier)
    - data: Data loading and domain clustering utilities
    - training: Four-phase training pipeline
    - evaluation: Attack evaluation metrics and analysis
    - utils: Helper functions and utilities

Reference:
"""

__version__ = "1.0.0"
__author__ = "Mosaic Research Team"

from . import models
from . import data
from . import training
from . import evaluation
from . import utils
