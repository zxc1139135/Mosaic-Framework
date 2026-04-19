"""
End-to-end membership inference on unseen query samples.
Combines target model querying, feature extraction, and attack network prediction.
"""

import logging
import numpy as np

from feature_extraction import extract_all_features, normalize_features
from memorization import compute_memorization_scores, compute_target_token_metadata
from metrics import compute_all_metrics, log_metrics, log_prediction_diagnostics, threshold_for_target_fpr

logger = logging.getLogger(__name__)


class MembershipInference:
    """End-to-end membership inference engine."""

    def __init__(
        self,
        target_model,
        reference_models,
        attack_trainer,
        tokenizer,
        max_length=512,
        batch_size=16,
        device="cuda",
        mu=1e-6,
        threshold=0.5,
        norm_method="none",
    ):
        self.target_model = target_model
        self.reference_models = reference_models
        self.attack_trainer = attack_trainer
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self.mu = mu
        self.threshold = threshold
        self.norm_method = norm_method

    def infer(self, query_texts, return_scores=True):
        logger.info("Inference on %d samples", len(query_texts))

        target_scores, _ = compute_memorization_scores(
            self.target_model,
            self.tokenizer,
            query_texts,
            self.max_length,
            self.batch_size,
            self.device,
        )
        target_token_meta = compute_target_token_metadata(
            self.target_model,
            self.tokenizer,
            query_texts,
            self.max_length,
            self.batch_size,
            self.device,
        )

        features = extract_all_features(
            self.target_model,
            self.reference_models,
            self.tokenizer,
            query_texts,
            target_scores,
            target_token_meta,
            self.max_length,
            self.batch_size,
            self.device,
            self.mu,
            expected_num_domains=len(self.reference_models),
        )
        features = normalize_features(features, method=self.norm_method)

        scores = self.attack_trainer.predict_scores(features)
        predictions = (scores >= self.threshold).astype(int)

        logger.info(
            "Inference threshold %.6f | predicted members=%d, predicted non-members=%d",
            self.threshold,
            int(predictions.sum()),
            int(len(predictions) - predictions.sum()),
        )
        result = {"predictions": predictions}
        if return_scores:
            result["scores"] = scores
        return result

    def evaluate(self, query_texts, true_labels):
        true_labels = np.array(true_labels)
        result = self.infer(query_texts, return_scores=True)
        metrics = compute_all_metrics(true_labels, result["scores"], threshold=self.threshold)
        log_prediction_diagnostics(true_labels, result["scores"], self.threshold, prefix="Evaluation")
        log_metrics(metrics, prefix="Evaluation")
        return metrics

    def calibrate_threshold(self, val_texts, val_labels, target_fpr=0.1):
        result = self.infer(val_texts, return_scores=True)
        self.threshold = threshold_for_target_fpr(np.array(val_labels), result["scores"], target_fpr=target_fpr)
        logger.info("Threshold calibrated to %.6f for target FPR %.4f", self.threshold, target_fpr)
        log_prediction_diagnostics(np.array(val_labels), result["scores"], self.threshold, prefix="Calibration")
        return self.threshold


def run_full_evaluation(target_model, reference_models, attack_trainer, tokenizer, eval_texts, eval_labels, config):
    engine = MembershipInference(
        target_model=target_model,
        reference_models=reference_models,
        attack_trainer=attack_trainer,
        tokenizer=tokenizer,
        max_length=config.model.max_seq_length,
        batch_size=config.distillation.batch_size // 2,
        device=config.device,
        mu=config.distillation.mu,
        threshold=config.attack.threshold,
        norm_method=config.attack.feature_norm,
    )
    return engine.evaluate(eval_texts, eval_labels)
