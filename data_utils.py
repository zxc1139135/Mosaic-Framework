"""
Data utilities for loading, preprocessing, and splitting datasets.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Generic text dataset with membership labels and offset mappings."""
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        if "offset_mapping" not in encoding:
            raise ValueError(
                f"Tokenizer {getattr(self.tokenizer, 'name_or_path', type(self.tokenizer).__name__)} "
                "does not provide offset_mapping. Use a fast tokenizer."
            )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "offset_mapping": encoding["offset_mapping"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "idx": torch.tensor(idx, dtype=torch.long),
            "text": self.texts[idx],
        }


class FeatureDataset(Dataset):
    """Dataset of extracted feature vectors with labels."""
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_wikimia(cache_dir: str) -> Tuple[List[str], List[str]]:
    ds = load_dataset("wjfu99/WikiMIA-24-perturbed", split="WikiMIA_length64", cache_dir=cache_dir)
    members = [r["input"] for r in ds if r["label"] == 1]
    nonmembers = [r["input"] for r in ds if r["label"] == 0]
    logger.info("WikiMIA: %d members, %d non-members", len(members), len(nonmembers))
    return members, nonmembers


def load_bookmia(cache_dir: str) -> Tuple[List[str], List[str]]:
    ds = load_dataset("swj0419/BookMIA", split="train", cache_dir=cache_dir)
    members = [r["text"] for r in ds if r["label"] == 1]
    nonmembers = [r["text"] for r in ds if r["label"] == 0]
    logger.info("BookMIA: %d members, %d non-members", len(members), len(nonmembers))
    return members, nonmembers


def load_pile(cache_dir: str, max_samples: int = 20000) -> Tuple[List[str], List[str]]:
    try:
        ds = load_dataset("monology/pile-uncopyrighted", split="train", streaming=True, cache_dir=cache_dir)
        texts = []
        for i, row in enumerate(ds):
            if i >= max_samples * 3:
                break
            text = row.get("text", "")
            if text:
                texts.append(text[:2048])
        if len(texts) < max_samples * 2:
            raise ValueError(f"Insufficient Pile samples collected: {len(texts)}")
        members = texts[:max_samples]
        nonmembers = texts[-max_samples:]
    except Exception as e:
        logger.warning("Pile load failed: %s. Using placeholder.", e)
        members = [f"Placeholder member sample {i}." for i in range(max_samples)]
        nonmembers = [f"Placeholder nonmember sample {i}." for i in range(max_samples)]
    logger.info("Pile: %d members, %d non-members", len(members), len(nonmembers))
    return members, nonmembers


def load_agnews(cache_dir: str) -> Tuple[List[str], List[str]]:
    ds = load_dataset("ag_news", cache_dir=cache_dir)
    members = [r["text"] for r in ds["train"]]
    nonmembers = [r["text"] for r in ds["test"]]
    logger.info("AG News: %d members, %d non-members", len(members), len(nonmembers))
    return members, nonmembers


DATASET_LOADERS = {
    "wikimia": load_wikimia,
    "bookmia": load_bookmia,
    "pile": load_pile,
    "agnews": load_agnews,
}


def _safe_slice_limit(total: int, start: int, desired: int) -> int:
    return max(0, min(desired, total - start))


def load_and_split_dataset(
    dataset_name: str,
    cache_dir: str,
    train_ratio: float = 0.8,
    ref_ratio: float = 0.1,
    eval_size: int = 2000,
    seed: int = 42,
) -> Dict:
    """
    Load dataset and create ref/eval splits.
    """
    rng = np.random.RandomState(seed)
    loader = DATASET_LOADERS.get(dataset_name)
    if loader is None:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    members, nonmembers = loader(cache_dir)
    members = list(members)
    nonmembers = list(nonmembers)
    rng.shuffle(members)
    rng.shuffle(nonmembers)

    n_m = len(members)
    n_nm = len(nonmembers)

    n_train_m = int(n_m * train_ratio)
    n_ref_m = _safe_slice_limit(n_m, n_train_m, int(n_m * ref_ratio))
    n_ref_nm = min(n_ref_m, n_nm // 2)

    eval_half = eval_size // 2
    n_eval_m = _safe_slice_limit(n_m, n_train_m + n_ref_m, eval_half)
    n_eval_nm = _safe_slice_limit(n_nm, n_ref_nm, eval_half)
    n_eval = min(n_eval_m, n_eval_nm)
    n_eval_m = n_eval
    n_eval_nm = n_eval

    ref_members = members[n_train_m:n_train_m + n_ref_m]
    eval_members = members[n_train_m + n_ref_m:n_train_m + n_ref_m + n_eval_m]

    ref_nonmembers = nonmembers[:n_ref_nm]
    eval_nonmembers = nonmembers[n_ref_nm:n_ref_nm + n_eval_nm]

    ref_texts = ref_members + ref_nonmembers
    ref_labels = [1] * len(ref_members) + [0] * len(ref_nonmembers)
    eval_texts = eval_members + eval_nonmembers
    eval_labels = [1] * len(eval_members) + [0] * len(eval_nonmembers)

    logger.info(
        "Splits - Ref: %d (members=%d, non-members=%d), Eval: %d (members=%d, non-members=%d)",
        len(ref_texts), len(ref_members), len(ref_nonmembers), len(eval_texts), len(eval_members), len(eval_nonmembers),
    )
    return {
        "ref_texts": ref_texts,
        "ref_labels": ref_labels,
        "eval_texts": eval_texts,
        "eval_labels": eval_labels,
    }


def stratified_batch_sampler(labels: np.ndarray, batch_size: int, seed: int = 42):
    rng = np.random.RandomState(seed)
    labels = np.asarray(labels)
    pos_idx = np.where(labels == 1)[0].copy()
    neg_idx = np.where(labels == 0)[0].copy()
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    half = max(batch_size // 2, 1)
    n_batches = min(len(pos_idx) // half, len(neg_idx) // half)
    for i in range(n_batches):
        batch = np.concatenate([pos_idx[i * half:(i + 1) * half], neg_idx[i * half:(i + 1) * half]])
        rng.shuffle(batch)
        yield batch


def stratified_train_val_split(labels: np.ndarray, val_ratio: float = 0.2, seed: int = 42):
    """Create a stratified train/validation split with both classes preserved."""
    labels = np.asarray(labels)
    rng = np.random.RandomState(seed)
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        raise ValueError(
            f"Need at least 2 samples per class for stratified split, got pos={len(pos_idx)}, neg={len(neg_idx)}"
        )

    pos_idx = pos_idx.copy()
    neg_idx = neg_idx.copy()
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    n_val_pos = max(1, int(round(len(pos_idx) * val_ratio)))
    n_val_neg = max(1, int(round(len(neg_idx) * val_ratio)))
    n_val_pos = min(n_val_pos, len(pos_idx) - 1)
    n_val_neg = min(n_val_neg, len(neg_idx) - 1)

    val_idx = np.concatenate([pos_idx[:n_val_pos], neg_idx[:n_val_neg]])
    train_idx = np.concatenate([pos_idx[n_val_pos:], neg_idx[n_val_neg:]])
    rng.shuffle(val_idx)
    rng.shuffle(train_idx)

    logger.info(
        "Attack split - Train: %d (members=%d, non-members=%d), Val: %d (members=%d, non-members=%d)",
        len(train_idx), int((labels[train_idx] == 1).sum()), int((labels[train_idx] == 0).sum()),
        len(val_idx), int((labels[val_idx] == 1).sum()), int((labels[val_idx] == 0).sum()),
    )
    return train_idx, val_idx
