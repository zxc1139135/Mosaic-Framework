"""Configuration module for the Mosaic framework."""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class ModelConfig:
    TARGET_MODELS = [
        "gpt2-medium",
        "facebook/opt-1.3b",
        "EleutherAI/pythia-1.4b",
        "meta-llama/Llama-2-7b-hf",
        "mistralai/Mistral-7B-v0.1",
    ]

    # Practical reduced-capacity defaults. For families without an official smaller
    # release in the exact same series, we use the closest openly available
    # tokenizer-compatible small variant and expose an override for strict experiments.
    REFERENCE_MODEL_MAP: Dict[str, str] = field(default_factory=lambda: {
        "gpt2-medium": "gpt2",
        "facebook/opt-1.3b": "facebook/opt-125m",
        "EleutherAI/pythia-1.4b": "EleutherAI/pythia-160m",
        "meta-llama/Llama-2-7b-hf": "TinyLlama/TinyLlama_v1.1",
        "mistralai/Mistral-7B-v0.1": "mistralai/Ministral-3-3B-Base-2512",
    })

    target_model: str = "gpt2-medium"
    reference_model_override: Optional[str] = None
    max_seq_length: int = 512

    def get_reference_model_name(self) -> str:
        if self.reference_model_override:
            return self.reference_model_override
        if self.target_model not in self.REFERENCE_MODEL_MAP:
            raise KeyError(f"No reference-model mapping found for target model: {self.target_model}")
        return self.REFERENCE_MODEL_MAP[self.target_model]


@dataclass
class DomainConfig:
    num_domains: int = 8
    member_ratio: float = 0.5
    strict_quantile_partition: bool = True


@dataclass
class DistillationConfig:
    epsilon: float = 0.3
    mu: float = 1e-6
    stage1_epochs: int = 15
    stage2_epochs: int = 15
    batch_size: int = 32
    weight_decay: float = 0.01
    lr_small: float = 5e-4
    lr_large: float = 5e-5
    lr_threshold_params: int = 1_000_000_000


@dataclass
class AttackNetworkConfig:
    hidden_dim: int = 256
    embed_dim: int = 128
    num_layers: int = 3
    epochs: int = 100
    batch_size: int = 256
    lr: float = 1e-3
    temperature: float = 0.07
    lambda_cls: float = 0.5
    threshold: float = 0.5
    feature_norm: str = "target_anchored"
    val_ratio: float = 0.2
    calibration_target_fpr: float = 0.1
    patience: int = 10


@dataclass
class DataConfig:
    SUPPORTED_DATASETS = ["wikimia", "bookmia", "pile", "agnews"]
    dataset_name: str = "wikimia"
    train_ratio: float = 0.8
    test_ratio: float = 0.1
    ref_ratio: float = 0.1
    eval_size: int = 2000
    data_dir: str = "./data"
    cache_dir: str = "./cache"


@dataclass
class ExperimentConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    domain: DomainConfig = field(default_factory=DomainConfig)
    distillation: DistillationConfig = field(default_factory=DistillationConfig)
    attack: AttackNetworkConfig = field(default_factory=AttackNetworkConfig)
    data: DataConfig = field(default_factory=DataConfig)
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "./outputs"
    num_workers: int = 4
    use_fp16: bool = False

    def get_feature_dim(self) -> int:
        return 1 + 5 * self.domain.num_domains

    def get_lr(self, num_params: int) -> float:
        return self.distillation.lr_large if num_params >= self.distillation.lr_threshold_params else self.distillation.lr_small

    def ensure_dirs(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data.data_dir, exist_ok=True)
        os.makedirs(self.data.cache_dir, exist_ok=True)
