"""Main entry point for the paper-aligned Mosaic framework."""

import argparse
import json
import logging
import os
import sys
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from baselines import run_baseline
from config import DataConfig, ExperimentConfig, ModelConfig
from data_utils import load_and_split_dataset, stratified_train_val_split
from feature_extraction import extract_all_features, normalize_features
from inference import run_full_evaluation
from memorization import compute_memorization_scores, compute_target_token_metadata, partition_into_domains
from metrics import compute_all_metrics, log_metrics, log_prediction_diagnostics, threshold_for_target_fpr
from reference_model import train_all_reference_models
from train import AttackNetworkTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("mosaic")

SELF_CONTAINED_BASELINES = {"zlib", "mink_pp", "neighborhood", "camia"}


def parse_args():
    p = argparse.ArgumentParser(description="Mosaic: Domain-Specialized MIA")
    p.add_argument("--target_model", type=str, default="gpt2-medium", choices=ModelConfig.TARGET_MODELS)
    p.add_argument("--reference_model", type=str, default=None, help="Override the default smaller reference-model family mapping.")
    p.add_argument("--dataset", type=str, default="wikimia", choices=DataConfig.SUPPORTED_DATASETS)
    p.add_argument("--max_seq_length", type=int, default=512)
    p.add_argument("--num_domains", type=int, default=8, help="Number of domains")
    p.add_argument("--epsilon", type=float, default=0.2)
    p.add_argument("--stage1_epochs", type=int, default=20)
    p.add_argument("--stage2_epochs", type=int, default=20)
    p.add_argument("--distill_batch_size", type=int, default=32)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--embed_dim", type=int, default=128)
    p.add_argument("--attack_epochs", type=int, default=100)
    p.add_argument("--attack_batch_size", type=int, default=256)
    p.add_argument("--attack_lr", type=float, default=1e-3)
    p.add_argument("--attack_val_ratio", type=float, default=0.2)
    p.add_argument("--calibration_target_fpr", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--lambda_cls", type=float, default=0.5)
    p.add_argument("--feature_norm", type=str, default="target_anchored", choices=["none", "target_anchored", "zscore", "minmax"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--output_dir", type=str, default="./outputs")
    p.add_argument("--cache_dir", type=str, default="./cache")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--use_fp16", action="store_true", default=False)
    p.add_argument("--no_fp16", action="store_true")
    p.add_argument("--run_baselines", action="store_true")
    p.add_argument("--baselines", nargs="+", default=["zlib", "mink_pp", "camia"])
    p.add_argument("--mode", type=str, default="full", choices=["full", "evaluate"])
    p.add_argument("--attack_ckpt", type=str, default=None)
    return p.parse_args()


def build_config(args):
    cfg = ExperimentConfig()
    cfg.model.target_model = args.target_model
    cfg.model.reference_model_override = args.reference_model
    cfg.model.max_seq_length = args.max_seq_length
    cfg.data.dataset_name = args.dataset
    cfg.data.cache_dir = args.cache_dir
    cfg.domain.num_domains = args.num_domains
    cfg.distillation.epsilon = args.epsilon
    cfg.distillation.stage1_epochs = args.stage1_epochs
    cfg.distillation.stage2_epochs = args.stage2_epochs
    cfg.distillation.batch_size = args.distill_batch_size
    cfg.attack.hidden_dim = args.hidden_dim
    cfg.attack.embed_dim = args.embed_dim
    cfg.attack.epochs = args.attack_epochs
    cfg.attack.batch_size = args.attack_batch_size
    cfg.attack.lr = args.attack_lr
    cfg.attack.val_ratio = args.attack_val_ratio
    cfg.attack.calibration_target_fpr = args.calibration_target_fpr
    cfg.attack.temperature = args.temperature
    cfg.attack.lambda_cls = args.lambda_cls
    cfg.attack.feature_norm = args.feature_norm
    cfg.seed = args.seed
    cfg.device = args.device
    cfg.output_dir = args.output_dir
    cfg.num_workers = args.num_workers
    cfg.use_fp16 = bool(args.use_fp16 and not args.no_fp16)
    cfg.ensure_dirs()
    return cfg


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_tokenizer(name: str, cache_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(name, cache_dir=cache_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_target_model(cfg):
    logger.info("Loading target model: %s", cfg.model.target_model)
    tokenizer = _load_tokenizer(cfg.model.target_model, cfg.data.cache_dir)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.target_model,
        cache_dir=cfg.data.cache_dir,
        torch_dtype=torch.float16 if cfg.use_fp16 else torch.float32,
    )
    model.eval().to(cfg.device)
    return model, tokenizer


def load_reference_models(cfg):
    base_dir = os.path.join(cfg.output_dir, "reference_models")
    reference_models = {}
    for domain_id in range(cfg.domain.num_domains):
        path = os.path.join(base_dir, f"ref_model_domain_{domain_id}")
        if not os.path.isdir(path):
            raise FileNotFoundError(f"Missing reference model directory: {path}")
        tokenizer = _load_tokenizer(path, cfg.data.cache_dir)
        model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16 if cfg.use_fp16 else torch.float32,
        ).to(cfg.device)
        model.eval()
        reference_models[domain_id] = {"model": model, "tokenizer": tokenizer, "name": path}
    logger.info("Loaded %d/%d reference models from %s", len(reference_models), cfg.domain.num_domains, base_dir)
    return reference_models


def log_domain_statistics(domains):
    logger.info("Domain statistics:")
    for domain_id, info in domains.items():
        logger.info(
            "  Domain %d: size=%d, members=%d, nonmembers=%d, score_range=[%.4f, %.4f]",
            domain_id,
            info["size"],
            info["n_members"],
            info["n_nonmembers"],
            info["boundary"][0],
            info["boundary"][1],
        )


def prepare_shared_artifacts(cfg, target_model, tokenizer, texts):
    mem_scores, token_lp = compute_memorization_scores(
        target_model,
        tokenizer,
        texts,
        cfg.model.max_seq_length,
        cfg.distillation.batch_size // 2,
        cfg.device,
    )
    target_token_meta = compute_target_token_metadata(
        target_model,
        tokenizer,
        texts,
        cfg.model.max_seq_length,
        cfg.distillation.batch_size // 2,
        cfg.device,
    )
    return mem_scores, token_lp, target_token_meta


def run_baselines_if_requested(args, cfg, target_model, tokenizer, eval_texts, eval_labels):
    baseline_results = {}
    if not args.run_baselines:
        return baseline_results

    logger.info("Running baselines...")
    eval_losses, eval_lp = compute_memorization_scores(
        target_model,
        tokenizer,
        eval_texts,
        cfg.model.max_seq_length,
        cfg.distillation.batch_size // 2,
        cfg.device,
    )
    for method in args.baselines:
        if method not in SELF_CONTAINED_BASELINES:
            logger.warning(
                "Skipping baseline %s in the default pipeline because it requires additional artifacts not produced by the core Mosaic training run.",
                method,
            )
            continue
        try:
            scores = run_baseline(
                method=method,
                model=target_model,
                tokenizer=tokenizer,
                texts=eval_texts,
                target_losses=eval_losses,
                token_logprobs=eval_lp,
                device=cfg.device,
            )
            bl_metrics = compute_all_metrics(eval_labels, scores)
            baseline_results[method] = bl_metrics
            log_metrics(bl_metrics, prefix=f"  [{method}]")
        except Exception as exc:
            logger.warning("Baseline %s failed: %s", method, exc)
    return baseline_results


def run_full_pipeline(args, cfg):
    t0 = time.time()
    target_model, tokenizer = load_target_model(cfg)
    data = load_and_split_dataset(
        cfg.data.dataset_name,
        cfg.data.cache_dir,
        cfg.data.train_ratio,
        cfg.data.ref_ratio,
        cfg.data.eval_size,
        cfg.seed,
    )
    ref_texts = data["ref_texts"]
    ref_labels = np.array(data["ref_labels"])
    eval_texts = data["eval_texts"]
    eval_labels = np.array(data["eval_labels"])

    logger.info("Phase I: Computing memorization scores and target token metadata...")
    mem_scores, _, target_token_meta = prepare_shared_artifacts(cfg, target_model, tokenizer, ref_texts)

    logger.info("Partitioning into %d domains with global score quantiles...", cfg.domain.num_domains)
    domains = partition_into_domains(mem_scores, ref_labels, cfg.domain.num_domains, strict=cfg.domain.strict_quantile_partition)
    log_domain_statistics(domains)

    ref_model_name = cfg.model.get_reference_model_name()
    logger.info("Training %d reference models (base: %s)...", cfg.domain.num_domains, ref_model_name)
    reference_models = train_all_reference_models(ref_model_name, domains, ref_texts, ref_labels.tolist(), target_token_meta, cfg)

    logger.info("Phase II: Extracting features...")
    train_features = extract_all_features(
        target_model,
        reference_models,
        tokenizer,
        ref_texts,
        mem_scores,
        target_token_meta,
        cfg.model.max_seq_length,
        cfg.distillation.batch_size // 2,
        cfg.device,
        cfg.distillation.mu,
        expected_num_domains=cfg.domain.num_domains,
    )
    train_features = normalize_features(train_features, method=cfg.attack.feature_norm)

    expected_dim = cfg.get_feature_dim()
    if train_features.shape[1] != expected_dim:
        raise ValueError(f"Feature dimension mismatch: got {train_features.shape[1]}, expected {expected_dim}")

    logger.info("Training contrastive attack network...")
    num_samples = len(ref_labels)
    if num_samples < 4:
        raise ValueError(f"At least 4 samples are required, got {num_samples}")

    train_idx, val_idx = stratified_train_val_split(ref_labels, val_ratio=cfg.attack.val_ratio, seed=cfg.seed)
    train_x, val_x = train_features[train_idx], train_features[val_idx]
    train_y, val_y = ref_labels[train_idx], ref_labels[val_idx]

    attack_trainer = AttackNetworkTrainer(
        input_dim=expected_dim,
        hidden_dim=cfg.attack.hidden_dim,
        embed_dim=cfg.attack.embed_dim,
        lr=cfg.attack.lr,
        temperature=cfg.attack.temperature,
        lambda_cls=cfg.attack.lambda_cls,
        epochs=cfg.attack.epochs,
        batch_size=cfg.attack.batch_size,
        device=cfg.device,
        patience=cfg.attack.patience,
    )
    attack_trainer.train(train_x, train_y, val_x, val_y)

    val_scores = attack_trainer.predict_scores(val_x)
    cfg.attack.threshold = threshold_for_target_fpr(val_y, val_scores, target_fpr=cfg.attack.calibration_target_fpr)
    logger.info("Calibrated decision threshold on held-out validation set: %.6f", cfg.attack.threshold)
    val_diag = log_prediction_diagnostics(val_y, val_scores, cfg.attack.threshold, prefix="Validation")

    attack_trainer.save(os.path.join(cfg.output_dir, "attack_network.pt"))

    logger.info("Evaluating on test set...")
    eval_metrics = run_full_evaluation(
        target_model,
        reference_models,
        attack_trainer,
        tokenizer,
        eval_texts,
        eval_labels.tolist(),
        cfg,
    )

    baseline_results = run_baselines_if_requested(args, cfg, target_model, tokenizer, eval_texts, eval_labels)

    elapsed = time.time() - t0
    results = {
        "config": {
            "target_model": cfg.model.target_model,
            "reference_model": cfg.model.get_reference_model_name(),
            "dataset": cfg.data.dataset_name,
            "num_domains": cfg.domain.num_domains,
            "epsilon": cfg.distillation.epsilon,
            "feature_dim": expected_dim,
            "feature_norm": cfg.attack.feature_norm,
            "threshold": float(cfg.attack.threshold),
            "use_fp16": cfg.use_fp16,
        },
        "validation_diagnostics": {k: float(v) for k, v in val_diag.items()},
        "mosaic_metrics": {k: float(v) for k, v in eval_metrics.items()},
        "baseline_metrics": {m: {k: float(v) for k, v in met.items()} for m, met in baseline_results.items()},
        "elapsed_seconds": elapsed,
    }
    with open(os.path.join(cfg.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info("=" * 60)
    logger.info("RESULTS SUMMARY")
    logger.info("Target: %s | Dataset: %s", cfg.model.target_model, cfg.data.dataset_name)
    logger.info("Domains=%d | feature_dim=%d | norm=%s", cfg.domain.num_domains, expected_dim, cfg.attack.feature_norm)
    log_metrics(eval_metrics, prefix="Mosaic")
    for method, met in baseline_results.items():
        log_metrics(met, prefix=f"  {method}")
    logger.info("Time: %.1fs (%.2fh)", elapsed, elapsed / 3600)
    return results


def run_evaluate_mode(args, cfg):
    if not args.attack_ckpt:
        logger.error("--attack_ckpt required for evaluate mode")
        sys.exit(1)

    target_model, tokenizer = load_target_model(cfg)
    reference_models = load_reference_models(cfg)

    ckpt = torch.load(args.attack_ckpt, map_location="cpu")
    ckpt_input_dim = ckpt.get("input_dim", cfg.get_feature_dim())
    if int(ckpt_input_dim) != cfg.get_feature_dim():
        raise ValueError(f"Checkpoint input_dim {ckpt_input_dim} does not match expected {cfg.get_feature_dim()}")

    attack_trainer = AttackNetworkTrainer(
        input_dim=cfg.get_feature_dim(),
        hidden_dim=cfg.attack.hidden_dim,
        embed_dim=cfg.attack.embed_dim,
        device=cfg.device,
    )
    attack_trainer.load(args.attack_ckpt)

    data = load_and_split_dataset(cfg.data.dataset_name, cfg.data.cache_dir, seed=cfg.seed)
    ref_texts = data["ref_texts"]
    ref_labels = np.array(data["ref_labels"])
    mem_scores, _, target_token_meta = prepare_shared_artifacts(cfg, target_model, tokenizer, ref_texts)
    ref_features = extract_all_features(
        target_model,
        reference_models,
        tokenizer,
        ref_texts,
        mem_scores,
        target_token_meta,
        cfg.model.max_seq_length,
        cfg.distillation.batch_size // 2,
        cfg.device,
        cfg.distillation.mu,
        expected_num_domains=cfg.domain.num_domains,
    )
    ref_features = normalize_features(ref_features, method=cfg.attack.feature_norm)
    _, val_idx = stratified_train_val_split(ref_labels, val_ratio=cfg.attack.val_ratio, seed=cfg.seed)
    val_scores = attack_trainer.predict_scores(ref_features[val_idx])
    cfg.attack.threshold = threshold_for_target_fpr(ref_labels[val_idx], val_scores, target_fpr=cfg.attack.calibration_target_fpr)
    log_prediction_diagnostics(ref_labels[val_idx], val_scores, cfg.attack.threshold, prefix="Evaluate/Validation")

    eval_metrics = run_full_evaluation(
        target_model,
        reference_models,
        attack_trainer,
        tokenizer,
        data["eval_texts"],
        data["eval_labels"],
        cfg,
    )
    log_metrics(eval_metrics, prefix="Evaluate")
    return eval_metrics


def main():
    args = parse_args()
    cfg = build_config(args)
    set_seed(cfg.seed)

    logger.info(
        "Mosaic | Mode: %s | Target: %s | Dataset: %s | K=%d | fp16=%s",
        args.mode,
        cfg.model.target_model,
        cfg.data.dataset_name,
        cfg.domain.num_domains,
        cfg.use_fp16,
    )

    if args.mode == "full":
        run_full_pipeline(args, cfg)
    elif args.mode == "evaluate":
        run_evaluate_mode(args, cfg)
    else:
        logger.error("Unsupported mode: %s", args.mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
