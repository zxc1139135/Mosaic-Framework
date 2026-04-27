"""
Evaluation metrics for membership inference attacks.
"""

import logging
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

logger = logging.getLogger(__name__)


def compute_auc(labels, scores):
    try:
        return roc_auc_score(labels, scores)
    except ValueError:
        return 0.5


def compute_tpr_at_fpr(labels, scores, target_fpr=0.1):
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(fpr, target_fpr, side="right") - 1
    idx = max(0, min(idx, len(fpr) - 2))
    fpr_lo, fpr_hi = fpr[idx], fpr[idx + 1]
    tpr_lo, tpr_hi = tpr[idx], tpr[idx + 1]
    if fpr_hi - fpr_lo < 1e-10:
        return float(tpr_lo)
    t = (target_fpr - fpr_lo) / (fpr_hi - fpr_lo)
    return float(tpr_lo + t * (tpr_hi - tpr_lo))


def find_optimal_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return float(thresholds[np.argmax(tpr - fpr)])


def threshold_for_target_fpr(labels, scores, target_fpr=0.1):
    fpr, _, thresholds = roc_curve(labels, scores)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return float(thresholds[np.argmin(fpr)])
    return float(thresholds[valid[-1]])


def compute_accuracy(labels, scores, threshold: Optional[float] = None):
    if threshold is None:
        threshold = find_optimal_threshold(labels, scores)
    return float(accuracy_score(labels, (scores >= threshold).astype(int)))


def compute_score_diagnostics(scores, threshold: Optional[float] = None) -> Dict[str, float]:
    scores = np.asarray(scores)
    if scores.size == 0:
        return {
            "ScoreMin": float("nan"),
            "ScoreMax": float("nan"),
            "ScoreMean": float("nan"),
            "ScoreMedian": float("nan"),
            "ScoreQ01": float("nan"),
            "ScoreQ05": float("nan"),
            "ScoreQ10": float("nan"),
            "ScoreQ90": float("nan"),
            "ScoreQ95": float("nan"),
            "ScoreQ99": float("nan"),
            "PredMemberRatio": float("nan"),
        }
    result = {
        "ScoreMin": float(scores.min()),
        "ScoreMax": float(scores.max()),
        "ScoreMean": float(scores.mean()),
        "ScoreMedian": float(np.median(scores)),
        "ScoreQ01": float(np.quantile(scores, 0.01)),
        "ScoreQ05": float(np.quantile(scores, 0.05)),
        "ScoreQ10": float(np.quantile(scores, 0.10)),
        "ScoreQ90": float(np.quantile(scores, 0.90)),
        "ScoreQ95": float(np.quantile(scores, 0.95)),
        "ScoreQ99": float(np.quantile(scores, 0.99)),
    }
    if threshold is not None:
        result["PredMemberRatio"] = float((scores >= threshold).mean())
    return result


def compute_all_metrics(labels, scores, fpr_targets=None, threshold: Optional[float] = None):
    if fpr_targets is None:
        fpr_targets = [0.001, 0.01, 0.05, 0.1, 0.2, 0.3]

    results = {
        "AUC": compute_auc(labels, scores),
        "Accuracy": compute_accuracy(labels, scores, threshold=threshold),
    }
    for fpr_val in fpr_targets:
        results[f"TPR@FPR={fpr_val}"] = compute_tpr_at_fpr(labels, scores, fpr_val)
    return results


def log_metrics(results, prefix=""):
    parts = [f"{k}: {v:.4f}" for k, v in results.items()]
    logger.info("%s %s", prefix, " | ".join(parts))


def log_prediction_diagnostics(true_labels, scores, threshold, prefix=""):
    true_labels = np.asarray(true_labels)
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)
    true_members = int((true_labels == 1).sum())
    true_nonmembers = int((true_labels == 0).sum())
    pred_members = int(preds.sum())
    pred_nonmembers = int(len(preds) - pred_members)
    diag = compute_score_diagnostics(scores, threshold=threshold)
    logger.info(
        "%s threshold=%.6f | true members=%d, true non-members=%d | predicted members=%d, predicted non-members=%d",
        prefix,
        threshold,
        true_members,
        true_nonmembers,
        pred_members,
        pred_nonmembers,
    )
    logger.info(
        "%s score stats | min=%.6f max=%.6f mean=%.6f median=%.6f q01=%.6f q05=%.6f q10=%.6f q90=%.6f q95=%.6f q99=%.6f pred_member_ratio=%.6f",
        prefix,
        diag["ScoreMin"],
        diag["ScoreMax"],
        diag["ScoreMean"],
        diag["ScoreMedian"],
        diag["ScoreQ01"],
        diag["ScoreQ05"],
        diag["ScoreQ10"],
        diag["ScoreQ90"],
        diag["ScoreQ95"],
        diag["ScoreQ99"],
        diag.get("PredMemberRatio", float("nan")),
    )
    return {
        "true_members": true_members,
        "true_nonmembers": true_nonmembers,
        "pred_members": pred_members,
        "pred_nonmembers": pred_nonmembers,
        **diag,
    }


def compute_per_domain_metrics(labels, scores, domain_assignments, num_domains):
    per_domain = {}
    for k in range(num_domains):
        mask = domain_assignments == k
        if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
            per_domain[k] = {"AUC": float("nan"), "n_samples": int(mask.sum())}
            continue
        per_domain[k] = compute_all_metrics(labels[mask], scores[mask], [0.1])
        per_domain[k]["n_samples"] = int(mask.sum())
    return per_domain
