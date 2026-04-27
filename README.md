# Not All Memories Are Equal: Unlocking the Potential of Heterogeneous Memorization in LLMs for Membership Inference

## Overview

**Mosaic** is a reference-based framework for membership inference attacks (MIAs) against large language models (LLMs). Unlike existing reference-based methods that treat all training samples uniformly, Mosaic is motivated by the observation that LLMs exhibit *heterogeneous memorization*, namely, different training samples are memorized to vastly different degrees. Uniform reference models fail to capture this variation and produce unreliable membership signals at the extremes of the memorization spectrum.

Mosaic addresses this by:

1. **Partitioning** the reference dataset into *K* memorization ranges based on per-sample memorization intensity.
2. **Training** a dedicated, lightweight reference model for each range via *differential distillation* — a two-stage sequential strategy that intentionally amplifies the behavioral divergence between members and non-members.
3. **Inferring** membership with a *contrastive attack model* that fuses cross-range behavioral features into a unified discriminative space, where members and non-members are naturally separated.

## Project Structure

```
Mosaic/
├── main.py               # Entry point: argument parsing, experiment orchestration
├── config.py             # All hyperparameters and model/dataset configurations
├── data_utils.py         # Dataset loading, splitting, and preprocessing
├── memorization.py       # Memorization score computation and quantile partitioning
├── reference_model.py    # Differential distillation, two-stage reference model training
├── feature_extraction.py # Cross-range feature extraction (Δ, Ω, Π, H̄, ρ̄)
├── token_alignment.py    # Token-level alignment utilities for distillation
├── train.py              # Contrastive attack model training (AttackNetworkTrainer)
├── attack_network.py     # MLP encoder + classification head architecture
├── inference.py          # Membership inference and threshold calibration
├── metrics.py            # AUC, accuracy, TPR@FPR evaluation utilities
├── baselines.py          # Implementations of comparison baselines
└── requirements.txt      # Python dependencies
```

---

## Requirements

```
torch>=2.1.0
transformers>=4.35.0
datasets>=2.14.0
numpy>=1.24.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

Install with:

```bash
pip install -r requirements.txt
```

## Supported Models and Datasets

### Target LLMs

| Model | HuggingFace ID |
|---|---|
| GPT-2-Medium | `gpt2-medium` |
| OPT-1.3B | `facebook/opt-1.3b` |
| Pythia-1.4B | `EleutherAI/pythia-1.4b` |
| LLaMA-2-7B | `meta-llama/Llama-2-7b-hf` |
| Mistral-7B | `mistralai/Mistral-7B-v0.1` |

Each target model is paired with a smaller reference model from the same family.

### Benchmark Datasets

| Dataset | Description |
|---|---|
| `wikimia` | Wikipedia-based MIA benchmark |
| `bookmia` | Books-based MIA benchmark |
| `pile` | The Pile (diverse web text) |
| `agnews` | AG News (news articles) |

## Quick Start

### Full Attack (train + evaluate)

```bash
python main.py \
    --target_model gpt2-medium \
    --dataset wikimia \
    --num_domains 8 \
    --epsilon 0.3 \
    --stage1_epochs 15 \
    --stage2_epochs 10 \
    --attack_epochs 100 \
    --output_dir ./outputs
```

### Evaluate Only (load checkpoint)

```bash
python main.py \
    --mode evaluate \
    --target_model meta-llama/Llama-2-7b-hf \
    --dataset bookmia \
    --attack_ckpt ./outputs/attack_model.pt
```

### Run with Baselines

```bash
python main.py \
    --target_model gpt2-medium \
    --dataset wikimia \
    --run_baselines \
    --baselines zlib mink_pp camia neighborhood
```

---

## Key Hyperparameters

| Parameter | Flag | Default | Description |
|---|---|---|---|
| Number of ranges | `--num_domains` | `8` | *K*: how many memorization ranges to partition into |
| Differential weight | `--epsilon` | `0.3` | Controls member vs. non-member loss weight asymmetry; ε ∈ (0, 0.5) |
| Stage 1 epochs | `--stage1_epochs` | `15` | Non-member distillation epochs per reference model |
| Stage 2 epochs | `--stage2_epochs` | `10` | Member distillation epochs per reference model |
| Contrastive temperature | `--temperature` | `0.07` | Temperature η in the supervised contrastive loss |
| Joint loss weight | `--lambda_cls` | `0.5` | λ: balance between contrastive loss and cross-entropy |
| Encoder hidden dim | `--hidden_dim` | `256` | MLP hidden dimension d_h |
| Embedding dim | `--embed_dim` | `128` | Output embedding dimension d |
| Feature normalization | `--feature_norm` | `target_anchored` | Normalization strategy: `none`, `target_anchored`, `zscore`, `minmax` |
| Calibration FPR | `--calibration_target_fpr` | `0.1` | Target FPR for threshold calibration (TPR@FPR metric) |

## Evaluation Metrics

- **AUC** — Area Under the ROC Curve (primary metric)
- **Attack Accuracy** — proportion of correct membership predictions
- **TPR@FPR=0.1** — true positive rate at a fixed false positive rate of 0.1, capturing performance at high-confidence operating points

## Baselines

The following MIA methods are re-implemented in `baselines.py` for direct comparison.
