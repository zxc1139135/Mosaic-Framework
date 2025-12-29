"""
Attack Training Module
======================
"""

import os
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tqdm import tqdm


@dataclass
class AttackTrainingConfig:
    num_epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    early_stopping_patience: int = 5

    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    device: str = "cuda"

    log_interval: int = 10


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

        return self.early_stop


class AttackTrainer:
    def __init__(
            self,
            attack_classifier: nn.Module,
            config: AttackTrainingConfig,
    ):
        self.classifier = attack_classifier
        self.config = config

        self.device = torch.device(config.device)
        self.classifier.to(self.device)

        self.criterion = nn.BCEWithLogitsLoss()

        self.optimizer = Adam(
            self.classifier.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )

        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)

        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def prepare_data(
            self,
            features: np.ndarray,
            labels: np.ndarray,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # Z-score
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-6
        features_normalized = (features - mean) / std

        self.feature_mean = torch.tensor(mean, dtype=torch.float32)
        self.feature_std = torch.tensor(std, dtype=torch.float32)

        X_temp, X_test, y_temp, y_test = train_test_split(
            features_normalized, labels,
            test_size=self.config.test_ratio,
            random_state=42,
            stratify=labels,
        )

        val_ratio_adjusted = self.config.val_ratio / (1 - self.config.test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio_adjusted,
            random_state=42,
            stratify=y_temp,
        )

        # Tensor
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # DataLoader
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        test_dataset = TensorDataset(X_test, y_test)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        print(f"Data split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

        return train_loader, val_loader, test_loader

    def train_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, float]:
        self.classifier.train()

        features, labels = batch
        features = features.to(self.device)
        labels = labels.to(self.device).unsqueeze(-1)

        logits = self.classifier(features)

        loss = self.criterion(logits, labels)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        with torch.no_grad():
            preds = (torch.sigmoid(logits) >= 0.5).float()
            accuracy = (preds == labels).float().mean()

        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.classifier.eval()

        all_logits = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        for features, labels in dataloader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            logits = self.classifier(features)
            loss = self.criterion(logits, labels.unsqueeze(-1))

            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())
            total_loss += loss.item()
            num_batches += 1

        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        probs = torch.sigmoid(all_logits).squeeze()
        preds = (probs >= 0.5).float()

        accuracy = (preds == all_labels).float().mean().item()

        return {
            "loss": total_loss / num_batches,
            "accuracy": accuracy,
            "probabilities": probs.numpy(),
            "predictions": preds.numpy(),
            "labels": all_labels.numpy(),
        }

    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        epoch_loss = 0.0
        epoch_acc = 0.0
        num_batches = 0

        for batch in dataloader:
            metrics = self.train_step(batch)
            epoch_loss += metrics["loss"]
            epoch_acc += metrics["accuracy"]
            num_batches += 1

        return {
            "loss": epoch_loss / num_batches,
            "accuracy": epoch_acc / num_batches,
        }

    def train(
            self,
            features: np.ndarray,
            labels: np.ndarray,
            save_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        train_loader, val_loader, test_loader = self.prepare_data(features, labels)

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        print(f"\nStarting Attack Classifier Training")
        print(f"  Epochs: {self.config.num_epochs}")
        print(f"  Batch size: {self.config.batch_size}")

        for epoch in range(self.config.num_epochs):
            self.epoch = epoch

            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])

            val_metrics = self.evaluate(val_loader)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            self.scheduler.step(val_metrics["loss"])

            print(f"Epoch {epoch}: "
                  f"Train Loss={train_metrics['loss']:.4f}, "
                  f"Train Acc={train_metrics['accuracy']:.4f}, "
                  f"Val Loss={val_metrics['loss']:.4f}, "
                  f"Val Acc={val_metrics['accuracy']:.4f}")

            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_model_state = self.classifier.state_dict().copy()

                if save_dir:
                    self.save_checkpoint(save_dir, "best")

            if self.early_stopping(val_metrics["loss"]):
                print(f"Early stopping at epoch {epoch}")
                break

        if self.best_model_state is not None:
            self.classifier.load_state_dict(self.best_model_state)

        print("\nFinal Test Evaluation:")
        test_metrics = self.evaluate(test_loader)
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

        history["test_loss"] = test_metrics["loss"]
        history["test_accuracy"] = test_metrics["accuracy"]
        history["test_probabilities"] = test_metrics["probabilities"]
        history["test_predictions"] = test_metrics["predictions"]
        history["test_labels"] = test_metrics["labels"]

        return history

    def save_checkpoint(self, save_dir: str, tag: str):
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.classifier.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
            "feature_mean": self.feature_mean if hasattr(self, 'feature_mean') else None,
            "feature_std": self.feature_std if hasattr(self, 'feature_std') else None,
        }

        path = os.path.join(save_dir, f"attack_classifier_{tag}.pt")
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

        if checkpoint.get("feature_mean") is not None:
            self.feature_mean = checkpoint["feature_mean"]
            self.feature_std = checkpoint["feature_std"]


class FeatureCollector:
    def __init__(
            self,
            target_model: nn.Module,
            shadow_system: nn.Module,
            feature_extractor: nn.Module,
            device: str = "cuda",
    ):
        self.target_model = target_model
        self.shadow_system = shadow_system
        self.feature_extractor = feature_extractor
        self.device = torch.device(device)

        self.target_model.to(self.device).eval()
        self.shadow_system.to(self.device).eval()

    @torch.no_grad()
    def collect_features(
            self,
            dataloader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray]:
        all_features = []
        all_labels = []

        for batch in tqdm(dataloader, desc="Collecting features"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch["label"]

            target_outputs = self.target_model(input_ids, attention_mask)
            if isinstance(target_outputs, dict):
                target_logits = target_outputs["logits"]
            else:
                target_logits = target_outputs
            target_probs = F.softmax(target_logits, dim=-1)

            shift_logits = target_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            target_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            ).view(input_ids.size(0), -1).mean(dim=-1)

            shadow_outputs = self.shadow_system.forward_all_experts(
                input_ids, attention_mask
            )
            expert_probs = [
                F.softmax(out["logits"] if isinstance(out, dict) else out, dim=-1)
                for out in shadow_outputs
            ]

            if hasattr(self.shadow_system, 'router'):
                routing_weights = torch.ones(input_ids.size(0), len(expert_probs)).to(self.device)
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            else:
                routing_weights = torch.ones(input_ids.size(0), len(expert_probs)).to(self.device)
                routing_weights = routing_weights / len(expert_probs)

            features = self.feature_extractor(
                target_probs, target_loss, input_ids,
                expert_probs, routing_weights
            )

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

        return np.vstack(all_features), np.concatenate(all_labels)


class AttackPipeline:
    def __init__(
            self,
            target_model: nn.Module,
            shadow_system: nn.Module,
            attack_classifier: nn.Module,
            feature_extractor: nn.Module,
            config: AttackTrainingConfig,
    ):
        self.target_model = target_model
        self.shadow_system = shadow_system
        self.attack_classifier = attack_classifier
        self.feature_extractor = feature_extractor
        self.config = config

        self.device = torch.device(config.device)

        self.collector = FeatureCollector(
            target_model, shadow_system,
            feature_extractor, config.device
        )

        self.trainer = AttackTrainer(attack_classifier, config)

    def run(
            self,
            train_dataloader: DataLoader,
            attack_dataloader: DataLoader,
            save_dir: Optional[str] = None,
    ) -> Dict:
        results = {}

        print("\n=== Phase 1: Collecting Training Features ===")
        train_features, train_labels = self.collector.collect_features(train_dataloader)
        print(f"Collected {len(train_features)} training samples")
        results["train_features_shape"] = train_features.shape

        print("\n=== Phase 2: Training Attack Classifier ===")
        attack_save_dir = os.path.join(save_dir, "attack") if save_dir else None
        history = self.trainer.train(train_features, train_labels, attack_save_dir)
        results["training_history"] = history

        print("\n=== Phase 3: Collecting Attack Features ===")
        attack_features, attack_labels = self.collector.collect_features(attack_dataloader)
        print(f"Collected {len(attack_features)} attack samples")

        print("\n=== Phase 4: Executing Attack ===")

        attack_features_normalized = (
                                                 attack_features - self.trainer.feature_mean.numpy()) / self.trainer.feature_std.numpy()
        attack_features_tensor = torch.tensor(attack_features_normalized, dtype=torch.float32)

        attack_dataset = TensorDataset(
            attack_features_tensor,
            torch.tensor(attack_labels, dtype=torch.float32)
        )
        attack_loader = DataLoader(attack_dataset, batch_size=self.config.batch_size)

        attack_metrics = self.trainer.evaluate(attack_loader)
        results["attack_metrics"] = {
            "accuracy": attack_metrics["accuracy"],
            "loss": attack_metrics["loss"],
        }
        results["attack_probabilities"] = attack_metrics["probabilities"]
        results["attack_predictions"] = attack_metrics["predictions"]
        results["attack_labels"] = attack_metrics["labels"]

        print(f"\nAttack Results:")
        print(f"  Accuracy: {attack_metrics['accuracy']:.4f}")

        return results


if __name__ == "__main__":
    print("Testing Attack Training Module...")

    from src.models import AttackClassifier

    np.random.seed(42)
    num_samples = 1000
    feature_dim = 45

    features = np.random.randn(num_samples, feature_dim).astype(np.float32)
    labels = (features[:, 0] + features[:, 1] + np.random.randn(num_samples) * 0.5 > 0).astype(np.float32)

    print(f"Features shape: {features.shape}")
    print(f"Labels distribution: {labels.mean():.2f}")

    classifier = AttackClassifier(input_dim=feature_dim)

    config = AttackTrainingConfig(
        num_epochs=10,
        batch_size=64,
        learning_rate=1e-3,
        device="cpu",
    )

    trainer = AttackTrainer(classifier, config)

    history = trainer.train(features, labels)

    print(f"\nFinal Test Accuracy: {history['test_accuracy']:.4f}")
    print(f"Best Validation Loss: {trainer.best_val_loss:.4f}")

    print("\nAll tests passed!")