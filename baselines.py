"""
Baseline membership inference attack methods.
"""

import zlib
import random
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Tuple
from tqdm import tqdm

logger = logging.getLogger(__name__)


@torch.no_grad()
def _get_token_logprobs(model, tokenizer, texts, max_length=512, batch_size=16, device="cuda"):
    """Compute per-sample loss and per-token log-probs."""
    model.eval()
    model.to(device)
    losses, all_lp = [], []
    for start in range(0, len(texts), batch_size):
        enc = tokenizer(texts[start:start+batch_size], truncation=True,
                        max_length=max_length, padding=True, return_tensors="pt").to(device)
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        logits = out.logits[:, :-1, :]
        labels = enc["input_ids"][:, 1:]
        mask = enc["attention_mask"][:, 1:].float()
        lp = torch.log_softmax(logits, dim=-1)
        tok_lp = torch.gather(lp, -1, labels.unsqueeze(-1)).squeeze(-1) * mask
        for i in range(tok_lp.size(0)):
            m = mask[i].bool()
            v = tok_lp[i][m].cpu().tolist()
            all_lp.append(v)
            losses.append(-sum(v) / max(len(v), 1))
    return np.array(losses), all_lp


def zlib_score(texts, target_losses):
    scores = []
    for i, text in enumerate(texts):
        zlib_bits = len(zlib.compress(text.encode("utf-8"))) * 8
        scores.append(-target_losses[i] / max(zlib_bits, 1))
    return np.array(scores)


def mink_pp_score(token_logprobs, k_percent=0.2):
    scores = []
    for lp_list in token_logprobs:
        if len(lp_list) == 0:
            scores.append(0.0)
            continue
        lp = np.array(lp_list)
        k = max(1, int(len(lp) * k_percent))
        bottom_k = np.sort(lp)[:k]
        mu, sigma = lp.mean(), lp.std() + 1e-8
        scores.append(((bottom_k - mu) / sigma).mean())
    return np.array(scores)


def neighborhood_score(model, tokenizer, texts, target_losses,
                       n_perturbations=10, mask_ratio=0.15,
                       max_length=512, device="cuda"):
    scores = []
    for i, text in tqdm(enumerate(texts), desc="Neighborhood", total=len(texts)):
        words = text.split()
        if len(words) < 3:
            scores.append(0.0)
            continue
        perturbed_losses = []
        for _ in range(n_perturbations):
            p = words.copy()
            n_mask = max(1, int(len(words) * mask_ratio))
            for pos in random.sample(range(len(words)), min(n_mask, len(words))):
                p[pos] = random.choice(words)
            loss, _ = _get_token_logprobs(model, tokenizer, [" ".join(p)],
                                          max_length, 1, device)
            perturbed_losses.append(loss[0])
        scores.append(np.mean(perturbed_losses) - target_losses[i])
    return np.array(scores)


def lira_score(target_losses, shadow_in_losses, shadow_out_losses):
    mu_in, sig_in = shadow_in_losses.mean(1), shadow_in_losses.std(1) + 1e-8
    mu_out, sig_out = shadow_out_losses.mean(1), shadow_out_losses.std(1) + 1e-8
    lp_in = -0.5 * ((target_losses - mu_in) / sig_in) ** 2 - np.log(sig_in)
    lp_out = -0.5 * ((target_losses - mu_out) / sig_out) ** 2 - np.log(sig_out)
    return lp_in - lp_out


def rmia_score(target_losses, ref_losses, pop_target, pop_ref, gamma=1.0):
    query_ratio = np.exp(-target_losses) / (np.exp(-ref_losses) + 1e-10)
    pop_ratio = np.exp(-pop_target) / (np.exp(-pop_ref) + 1e-10)
    return np.array([np.mean(pop_ratio < gamma * qr) for qr in query_ratio])


def camia_score(token_logprobs, k_percent=0.2):
    scores = []
    for lp_list in token_logprobs:
        if len(lp_list) < 2:
            scores.append(0.0)
            continue
        lp = np.array(lp_list)
        k = max(1, int(len(lp) * k_percent))
        s1 = np.sort(lp)[:k].mean()
        s2 = -lp.var()
        s3 = lp.min() / (lp.max() + 1e-8)
        scores.append(s1 - 0.5 * s2 + 0.3 * s3)
    return np.array(scores)


def con_recall_score(model, tokenizer, texts, member_prefixes, nonmember_prefixes,
                     max_length=512, batch_size=8, device="cuda"):
    base_losses, _ = _get_token_logprobs(model, tokenizer, texts, max_length, batch_size, device)
    n_pfx = min(5, len(member_prefixes), len(nonmember_prefixes))
    m_shift = np.zeros(len(texts))
    nm_shift = np.zeros(len(texts))
    for pfx in member_prefixes[:n_pfx]:
        l, _ = _get_token_logprobs(model, tokenizer, [pfx + " " + t for t in texts],
                                    max_length, batch_size, device)
        m_shift += (l - base_losses) / n_pfx
    for pfx in nonmember_prefixes[:n_pfx]:
        l, _ = _get_token_logprobs(model, tokenizer, [pfx + " " + t for t in texts],
                                    max_length, batch_size, device)
        nm_shift += (l - base_losses) / n_pfx
    return nm_shift - m_shift


def icp_mia_score(model, tokenizer, texts, icl_examples,
                  max_length=512, batch_size=8, device="cuda", ref_model=None):
    base, _ = _get_token_logprobs(model, tokenizer, texts, max_length, batch_size, device)
    prefix = " ".join(icl_examples[:3])
    cond, _ = _get_token_logprobs(model, tokenizer, [prefix + " " + t for t in texts],
                                   max_length, batch_size, device)
    score = base - cond
    if ref_model is not None:
        rb, _ = _get_token_logprobs(ref_model, tokenizer, texts, max_length, batch_size, device)
        rc, _ = _get_token_logprobs(ref_model, tokenizer, [prefix + " " + t for t in texts],
                                     max_length, batch_size, device)
        score = score - (rb - rc)
    return score


BASELINE_METHODS = ["zlib", "mink_pp", "neighborhood", "lira", "rmia",
                    "camia", "con_recall", "icp_mia"]


def run_baseline(method, model, tokenizer, texts, target_losses, token_logprobs,
                 device="cuda", **kwargs):
    logger.info(f"Running baseline: {method}")
    if method == "zlib":
        return zlib_score(texts, target_losses)
    elif method == "mink_pp":
        return mink_pp_score(token_logprobs, kwargs.get("k_percent", 0.2))
    elif method == "neighborhood":
        return neighborhood_score(model, tokenizer, texts, target_losses, device=device)
    elif method == "lira":
        return lira_score(target_losses, kwargs["shadow_in_losses"], kwargs["shadow_out_losses"])
    elif method == "rmia":
        return rmia_score(target_losses, kwargs["ref_losses"],
                          kwargs["population_target_losses"], kwargs["population_ref_losses"])
    elif method == "camia":
        return camia_score(token_logprobs)
    elif method == "con_recall":
        return con_recall_score(model, tokenizer, texts,
                                kwargs["member_prefixes"], kwargs["nonmember_prefixes"],
                                device=device)
    elif method == "icp_mia":
        return icp_mia_score(model, tokenizer, texts,
                             kwargs.get("icl_examples", texts[:5]),
                             device=device, ref_model=kwargs.get("ref_model"))
    else:
        raise ValueError(f"Unknown baseline: {method}")
