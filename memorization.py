"""
Memorization score computation and paper-aligned domain partitioning.

S(x) = -1 / |x| * sum(log P(w_i | w_<i)) for each sample.
Lower S(x) indicates stronger memorization.

Samples are partitioned globally by memorization score into K consecutive
quantile domains, matching the paper description.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


def _tokenize_with_offsets(tokenizer, texts: List[str], max_length: int):
    enc = tokenizer(
        texts,
        truncation=True,
        max_length=max_length,
        padding=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    if "offset_mapping" not in enc:
        raise ValueError(
            f"Tokenizer {getattr(tokenizer, 'name_or_path', type(tokenizer).__name__)} "
            "does not provide offset_mapping. Please use a fast tokenizer."
        )
    return enc


@torch.no_grad()
def compute_memorization_scores(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    batch_size: int = 16,
    device: str = "cuda",
) -> Tuple[np.ndarray, List[List[float]]]:
    """
    Compute average negative log-likelihood per sample.

    Lower score means stronger memorization.

    Returns:
        scores: shape (N,)
        token_logprobs: per-token log-probabilities for each sample
    """
    model.eval()
    model.to(device)

    scores = []
    all_token_lp = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Memorization scores"):
        batch_texts = texts[start:start + batch_size]
        enc = _tokenize_with_offsets(tokenizer, batch_texts, max_length)
        offset_mapping = enc.pop("offset_mapping")
        enc = enc.to(device)

        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        logits = model(input_ids=ids, attention_mask=mask).logits

        s_logits = logits[:, :-1, :]
        s_labels = ids[:, 1:]
        s_mask = mask[:, 1:].float()

        log_probs = torch.log_softmax(s_logits, dim=-1)
        tok_lp = torch.gather(log_probs, -1, s_labels.unsqueeze(-1)).squeeze(-1)
        tok_lp = tok_lp * s_mask

        for i in range(tok_lp.size(0)):
            valid_mask = s_mask[i].bool()
            valid_log_probs = tok_lp[i][valid_mask].cpu().tolist()
            all_token_lp.append(valid_log_probs)

            num_tokens = valid_mask.sum().item()
            score = -sum(valid_log_probs) / num_tokens if num_tokens > 0 else float("inf")
            scores.append(score)

    return np.array(scores), all_token_lp


@torch.no_grad()
def compute_target_token_metadata(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    batch_size: int = 16,
    device: str = "cuda",
) -> List[Dict]:
    """Compute target token probabilities, log-probabilities, and character offsets."""
    model.eval()
    model.to(device)

    metadata: List[Dict] = []

    for start in tqdm(range(0, len(texts), batch_size), desc="Target token metadata"):
        batch_texts = texts[start:start + batch_size]
        enc = _tokenize_with_offsets(tokenizer, batch_texts, max_length)
        offset_mapping = enc.pop("offset_mapping")
        enc = enc.to(device)

        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        logits = model(input_ids=ids, attention_mask=mask).logits

        s_logits = logits[:, :-1, :]
        s_labels = ids[:, 1:]
        s_mask = mask[:, 1:].float()

        probs = torch.softmax(s_logits, dim=-1)
        log_probs = torch.log_softmax(s_logits, dim=-1)
        tok_probs = torch.gather(probs, -1, s_labels.unsqueeze(-1)).squeeze(-1)
        tok_log_probs = torch.gather(log_probs, -1, s_labels.unsqueeze(-1)).squeeze(-1)

        for i, text in enumerate(batch_texts):
            valid_mask = s_mask[i].bool().cpu()
            # Shift offset_mapping by one to match the predicted next-token labels.
            shifted_offsets = offset_mapping[i][1:]
            valid_offsets = shifted_offsets[valid_mask].tolist()
            metadata.append(
                {
                    "text": text,
                    "text_length": len(text),
                    "token_probs": tok_probs[i][valid_mask.to(tok_probs.device)].detach().cpu().tolist(),
                    "token_logprobs": tok_log_probs[i][valid_mask.to(tok_log_probs.device)].detach().cpu().tolist(),
                    "offsets": valid_offsets,
                }
            )

    return metadata


@torch.no_grad()
def compute_target_token_probs(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 512,
    batch_size: int = 16,
    device: str = "cuda",
) -> List[List[float]]:
    """Backward-compatible wrapper returning token-level target probabilities only."""
    metadata = compute_target_token_metadata(model, tokenizer, texts, max_length, batch_size, device)
    return [item["token_probs"] for item in metadata]


def _split_sorted_indices_evenly(sorted_indices: np.ndarray, num_domains: int) -> List[np.ndarray]:
    """Split a sorted index array into K consecutive nearly equal chunks."""
    total = len(sorted_indices)
    base = total // num_domains
    remainder = total % num_domains

    chunks = []
    offset = 0
    for k in range(num_domains):
        chunk_size = base + (1 if k < remainder else 0)
        chunk = sorted_indices[offset:offset + chunk_size]
        chunks.append(chunk)
        offset += chunk_size
    return chunks


def partition_into_domains(
    scores: np.ndarray,
    labels: np.ndarray,
    num_domains: int = 8,
    strict: bool = True,
) -> Dict[int, Dict]:
    """
    Partition all samples globally into K consecutive score-quantile domains.

    This matches the paper description: all samples are sorted together by S(x),
    then split into K nearly equal-sized consecutive subsets.

    If strict=True, every domain must contain at least one member and one
    non-member; otherwise a ValueError is raised rather than silently changing
    the method or skipping domains.
    """
    if len(scores) != len(labels):
        raise ValueError(
            f"scores and labels must have the same length, got {len(scores)} and {len(labels)}"
        )
    if num_domains <= 0:
        raise ValueError(f"num_domains must be positive, got {num_domains}")

    labels = np.asarray(labels)
    scores = np.asarray(scores)
    if len(np.unique(labels)) < 2:
        raise ValueError("Both member and non-member samples are required for domain partitioning.")

    sorted_idx = np.argsort(scores)
    chunks = _split_sorted_indices_evenly(sorted_idx, num_domains)
    domains: Dict[int, Dict] = {}

    for k, chunk in enumerate(chunks):
        chunk = np.asarray(chunk, dtype=int)
        member_idx = chunk[labels[chunk] == 1]
        nonmember_idx = chunk[labels[chunk] == 0]

        if len(chunk) > 0:
            domain_low = float(scores[chunk].min())
            domain_high = float(scores[chunk].max())
        else:
            domain_low = float("nan")
            domain_high = float("nan")

        domains[k] = {
            "indices": chunk,
            "member_indices": member_idx,
            "nonmember_indices": nonmember_idx,
            "boundary": (domain_low, domain_high),
            "size": int(len(chunk)),
            "n_members": int(len(member_idx)),
            "n_nonmembers": int(len(nonmember_idx)),
        }

        logger.info(
            "Domain %d: S in [%.3f, %.3f], n=%d, m=%d, nm=%d",
            k,
            domain_low,
            domain_high,
            len(chunk),
            len(member_idx),
            len(nonmember_idx),
        )

        if strict and (len(member_idx) == 0 or len(nonmember_idx) == 0):
            raise ValueError(
                "Paper-aligned global quantile partition produced an invalid domain "
                f"{k} with m={len(member_idx)} and nm={len(nonmember_idx)}. "
                "Increase data size, reduce K, or change the split; the code will not "
                "silently skip the domain because that would change the method."
            )

    return domains
