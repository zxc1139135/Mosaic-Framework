# Mosaic Framework

A mixture-of-experts approach for membership inference attacks on large language models.

## Overview

Mosaic implements a domain-specialized expert system for conducting membership inference attacks against LLMs. The framework leverages multiple expert models to capture heterogeneous memorization patterns across different data domains.

### Key Features

- Four-stage training pipeline: expert construction, domain training, ensemble integration, attack execution
- Multi-expert system with configurable number of specialized models
- Dynamic routing with top-k sparse activation
- Multi-granularity evaluation support

## Project Structure

```
mosaic/
├── configs/
│   └── config.yaml
├── src/
│   ├── models/
│   │   ├── expert_model.py
│   │   ├── router_network.py
│   │   ├── meta_learner.py
│   │   └── attack_classifier.py
│   ├── data/
│   │   ├── data_loader.py
│   │   └── domain_clustering.py
│   ├── training/
│   │   ├── expert_trainer.py
│   │   ├── ensemble_trainer.py
│   │   └── attack_trainer.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── utils/
│       └── helpers.py
├── scripts/
│   ├── train.py
│   └── demo.py
├── tests/
│   └── test_all.py
└── requirements.txt
```

## Getting Started

### Installation

```bash
git clone <repository_url>
cd mosaic

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

### Training

```bash
python scripts/train.py --config configs/config.yaml

python scripts/train.py --config configs/config.yaml --phase 2

python scripts/train.py --config configs/config.yaml --output outputs/experiment
```

### Testing

```bash
python tests/test_all.py
```

## Configuration

```yaml
expert_system:
  num_experts: 8
  expert_architecture_small:
    num_layers: 6
    hidden_dim: 768
    num_attention_heads: 12

domain_training:
  num_epochs: 160
  loss_weights:
    alpha: 0.6
    beta: 0.3
    gamma: 0.1
    delta: 0.1

ensemble_training:
  top_k_experts: 3
  
attack_execution:
  feature_dims:
    total: 45
```

## Evaluation Metrics

| Metric | Description |
|--------|-------------|
| AUC-ROC | Area under ROC curve |
| AUPRC | Area under precision-recall curve |
| TPR@FPR | True positive rate at given false positive rate |
| Attack Advantage | Maximum TPR - FPR |

## Supported Target Models

- GPT-2 (124M, 355M)
- GPT-Neo (1.3B)
- OPT (1.3B)
- LLaMA-2 (7B)
- Pythia (1.4B)

## License

MIT License

## Disclaimer

This framework is for research purposes only.
