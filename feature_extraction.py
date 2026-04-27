"""
Reference feature extraction module.
"""

import logging
from typing import Dict, List, Union

import numpy as np
import torch
from tqdm import tqdm

from token_alignment import mean_abs_discrepancy_over_text

logger = logging.getLogger(__name__)


@torch.no_grad()
def extract_ref_outputs(model, tokenizer, texts, max_length=512, batch_size=16, device="cuda"):
    """Forward texts through a reference model and extract statistics."""
    model.eval()
    model.to(device)
    all_loss, all_lp, all_ent, all_top1, all_offsets = [], [], [], [], []

    for start in tqdm(range(0, len(texts), batch_size), desc="Ref features", leave=False):
        batch_texts = texts[start:start + batch_size]
        enc = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )
        if "offset_mapping" not in enc:
            raise ValueError(
                f"Tokenizer {getattr(tokenizer, 'name_or_path', type(tokenizer).__name__)} does not provide offset_mapping. Use a fast tokenizer."
            )
        offset_mapping = enc.pop("offset_mapping")
        enc = enc.to(device)

        ids, mask = enc["input_ids"], enc["attention_mask"]
        logits = model(input_ids=ids, attention_mask=mask).logits

        s_logits = logits[:, :-1, :]
        s_labels = ids[:, 1:]
        s_mask = mask[:, 1:].float()

        log_probs = torch.log_softmax(s_logits, dim=-1)
        tok_lp = torch.gather(log_probs, -1, s_labels.unsqueeze(-1)).squeeze(-1)

        probs = torch.softmax(s_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        top1 = probs.max(dim=-1).values

        for i in range(ids.size(0)):
            m = s_mask[i].bool()
            n = m.sum().item()
            valid_offsets = offset_mapping[i][1:][m.cpu()].tolist()
            all_offsets.append(valid_offsets)
            if n > 0:
                valid_lp = tok_lp[i][m].cpu().tolist()
                all_lp.append(valid_lp)
                all_loss.append(-sum(valid_lp) / n)
                all_ent.append(entropy[i][m].mean().item())
                all_top1.append(top1[i][m].mean().item())
            else:
                all_lp.append([])
                all_loss.append(float("inf"))
                all_ent.append(0.0)
                all_top1.append(0.0)

    return {
        "loss": np.array(all_loss),
        "token_logprobs": all_lp,
        "offsets": all_offsets,
        "entropy": np.array(all_ent),
        "top1_prob": np.array(all_top1),
    }


def compute_domain_features(target_loss, target_token_meta: List[Dict], ref_outputs, mu=1e-6):
    """Compute the 5-dim feature vector for one reference model."""
    n = len(target_loss)
    ref_loss = ref_outputs["loss"]
    ref_lp = ref_outputs["token_logprobs"]
    ref_offsets = ref_outputs["offsets"]

    delta = ref_loss - target_loss
    omega = ref_loss / (target_loss + mu)

    pi = np.zeros(n, dtype=np.float32)
    for i in range(n):
        meta = target_token_meta[i]
        pi[i] = mean_abs_discrepancy_over_text(
            text_length=meta.get("text_length", len(meta.get("text", ""))),
            offsets_a=meta["offsets"],
            values_a=meta["token_logprobs"],
            offsets_b=ref_offsets[i],
            values_b=ref_lp[i],
        )

    return np.stack([delta, omega, pi, ref_outputs["entropy"], ref_outputs["top1_prob"]], axis=1)


def _resolve_ref_model_and_tokenizer(entry: Union[dict, torch.nn.Module], fallback_tokenizer):
    if isinstance(entry, dict):
        model = entry["model"]
        tokenizer = entry.get("tokenizer", fallback_tokenizer)
        name = entry.get("name", getattr(model, "name_or_path", "reference_model"))
    else:
        model = entry
        tokenizer = fallback_tokenizer
        name = getattr(model, "name_or_path", "reference_model")
    return model, tokenizer, name


def extract_all_features(
    target_model,
    reference_models: Dict[int, Union[dict, torch.nn.Module]],
    tokenizer,
    texts,
    target_loss,
    target_token_meta,
    max_length=512,
    batch_size=16,
    device="cuda",
    mu=1e-6,
    expected_num_domains=None,
):
    """
    Extract f(x) = [f_T || f_1 || ... || f_K] for all samples.
    """
    n = len(texts)
    if expected_num_domains is None:
        expected_num_domains = len(reference_models)

    expected_ids = list(range(expected_num_domains))
    actual_ids = sorted(reference_models.keys())
    if actual_ids != expected_ids:
        raise ValueError(f"Reference model ids must be exactly {expected_ids}, got {actual_ids}.")

    features = np.zeros((n, 1 + 5 * expected_num_domains), dtype=np.float32)
    features[:, 0] = target_loss

    for dom_id in expected_ids:
        ref_model, ref_tokenizer, ref_name = _resolve_ref_model_and_tokenizer(reference_models[dom_id], tokenizer)
        logger.info(
            "Extracting features for ref model %d/%d (%s)",
            dom_id + 1,
            expected_num_domains,
            ref_name,
        )
        ref_out = extract_ref_outputs(
            ref_model,
            ref_tokenizer,
            texts,
            max_length,
            batch_size,
            device,
        )
        dom_feats = compute_domain_features(target_loss, target_token_meta, ref_out, mu)
        col = 1 + dom_id * 5
        features[:, col:col + 5] = dom_feats

    logger.info("Feature extraction done. Shape: %s", tuple(features.shape))
    return features


def normalize_features(features, method="none"):
    """
    Normalize features.
    """
    if method == "none":
        return features

    features = features.copy()
    if method == "target_anchored":
        anchor = np.clip(features[:, 0:1], 1e-6, None)
        for k in range((features.shape[1] - 1) // 5):
            b = 1 + k * 5
            features[:, b] /= anchor.squeeze()
            features[:, b + 2] /= anchor.squeeze()
        mu, sigma = features.mean(0), features.std(0) + 1e-8
        features = (features - mu) / sigma
    elif method == "zscore":
        mu, sigma = features.mean(0), features.std(0) + 1e-8
        features = (features - mu) / sigma
    elif method == "minmax":
        fmin, fmax = features.min(0), features.max(0)
        features = (features - fmin) / (fmax - fmin + 1e-8)
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    return features
