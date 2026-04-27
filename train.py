"""
Training pipeline for the contrastive attack model.
"""

import os
import logging
import numpy as np
import torch
from torch.optim import Adam
from typing import Dict

from data_utils import stratified_batch_sampler
from attack_network import ContrastiveAttackNetwork, AttackLoss
from metrics import compute_all_metrics

logger = logging.getLogger(__name__)


class AttackNetworkTrainer:
    """Trainer for the contrastive attack model."""
    def __init__(
        self,
        input_dim,
        hidden_dim=256,
        embed_dim=128,
        lr=1e-3,
        temperature=0.07,
        lambda_cls=0.5,
        epochs=100,
        batch_size=256,
        device="cuda",
        patience=10,
    ):
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.input_dim = int(input_dim)

        self.network = ContrastiveAttackNetwork(
            self.input_dim,
            hidden_dim,
            embed_dim,
        ).to(device)

        self.criterion = AttackLoss(temperature, lambda_cls)
        self.optimizer = Adam(self.network.parameters(), lr=lr)

    def _validate_features(self, features, name="features"):
        if features is None:
            raise ValueError(f"{name} is None")
        if not hasattr(features, "shape"):
            raise TypeError(f"{name} must have .shape, got {type(features)}")
        if len(features.shape) != 2:
            raise ValueError(f"{name} must be 2D, got shape={features.shape}")
        if features.shape[0] == 0:
            raise ValueError(f"{name} is empty, got shape={features.shape}")
        if features.shape[1] != self.input_dim:
            raise ValueError(
                f"{name} feature dim mismatch: got {features.shape[1]}, "
                f"expected {self.input_dim}. "
                f"This usually indicates inconsistent feature extraction "
                f"between train/val/test."
            )

    def train(self, train_features, train_labels,
              val_features=None, val_labels=None) -> Dict[str, list]:
        self._validate_features(train_features, "train_features")
        if val_features is not None:
            self._validate_features(val_features, "val_features")

        if len(train_features) != len(train_labels):
            raise ValueError(
                f"train_features/train_labels length mismatch: "
                f"{len(train_features)} vs {len(train_labels)}"
            )

        if val_features is not None and val_labels is not None:
            if len(val_features) != len(val_labels):
                raise ValueError(
                    f"val_features/val_labels length mismatch: "
                    f"{len(val_features)} vs {len(val_labels)}"
                )

        history = {"train_loss": [], "val_auc": []}
        best_val_auc = 0.0
        best_state = None
        patience_ctr = 0

        logger.info(
            f"Training attack network: {len(train_labels)} samples, "
            f"{self.epochs} epochs, input_dim={self.input_dim}"
        )

        for epoch in range(self.epochs):
            self.network.train()
            epoch_loss = 0.0
            n_batches = 0

            for batch_idx in stratified_batch_sampler(
                train_labels,
                self.batch_size,
                seed=epoch,
            ):
                feats = torch.tensor(
                    train_features[batch_idx],
                    dtype=torch.float32,
                    device=self.device,
                )
                labs = torch.tensor(
                    train_labels[batch_idx],
                    dtype=torch.long,
                    device=self.device,
                )

                if feats.ndim != 2 or feats.shape[1] != self.input_dim:
                    raise ValueError(
                        f"Batch feature dim mismatch at epoch {epoch + 1}: "
                        f"got {tuple(feats.shape)}, expected (*, {self.input_dim})"
                    )

                self.optimizer.zero_grad()
                emb, probs = self.network(feats)
                total_loss, l_con, l_cls = self.criterion(emb, probs, labs)
                total_loss.backward()
                self.optimizer.step()

                epoch_loss += total_loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_loss)

            if val_features is not None and val_labels is not None:
                val_scores = self.predict_scores(val_features)
                val_auc = compute_all_metrics(val_labels, val_scores, [0.1])["AUC"]
                history["val_auc"].append(val_auc)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_state = {
                        k: v.detach().cpu().clone()
                        for k, v in self.network.state_dict().items()
                    }
                    patience_ctr = 0
                else:
                    patience_ctr += 1

                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{self.epochs} - "
                        f"Loss: {avg_loss:.4f} - Val AUC: {val_auc:.4f}"
                    )

                if patience_ctr >= self.patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{self.epochs} - Loss: {avg_loss:.4f}"
                    )

        if best_state is not None:
            self.network.load_state_dict(best_state)
            logger.info(f"Restored best model (Val AUC: {best_val_auc:.4f})")

        return history

    @torch.no_grad()
    def predict_scores(self, features):
        """Predict membership scores for a feature matrix."""
        self._validate_features(features, "predict_features")

        self.network.eval()
        all_scores = []

        for start in range(0, len(features), self.batch_size):
            batch_np = features[start:start + self.batch_size]

            if batch_np.shape[1] != self.input_dim:
                raise ValueError(
                    f"Predict batch feature dim mismatch: got {batch_np.shape[1]}, "
                    f"expected {self.input_dim}"
                )

            batch = torch.tensor(
                batch_np,
                dtype=torch.float32,
                device=self.device,
            )
            _, probs = self.network(batch)
            all_scores.append(probs.cpu().numpy())

        if len(all_scores) == 0:
            return np.array([], dtype=np.float32)

        return np.concatenate(all_scores)

    @torch.no_grad()
    def predict(self, features, threshold=0.5):
        """Return binary membership predictions."""
        return (self.predict_scores(features) >= threshold).astype(int)

    def save(self, path):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(
            {
                "network": self.network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "input_dim": self.input_dim,
            },
            path,
        )
        logger.info(f"Attack network saved to {path} (input_dim={self.input_dim})")

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)

        ckpt_input_dim = ckpt.get("input_dim", None)
        if ckpt_input_dim is not None and int(ckpt_input_dim) != self.input_dim:
            raise ValueError(
                f"Checkpoint input_dim mismatch: ckpt={ckpt_input_dim}, "
                f"current_trainer={self.input_dim}"
            )

        self.network.load_state_dict(ckpt["network"])

        if "optimizer" in ckpt and ckpt["optimizer"] is not None:
            try:
                self.optimizer.load_state_dict(ckpt["optimizer"])
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")

        logger.info(f"Attack network loaded from {path}")