"""
Ensemble Training Module
========================

"""

import os
from typing import Optional, Tuple, Dict, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


@dataclass
class EnsembleTrainingConfig:
    num_epochs: int = 50
    batch_size: int = 64
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    lambda_balance: float = 0.1
    lambda_explore: float = 0.05
    lambda_reg: float = 1e-4

    top_k_experts: int = 3

    device: str = "cuda"

    log_interval: int = 50
    save_interval: int = 5


class RouterLoss(nn.Module):
    def __init__(
            self,
            lambda_balance: float = 0.1,
            lambda_explore: float = 0.05,
            num_experts: int = 8,
    ):
        super().__init__()
        self.lambda_balance = lambda_balance
        self.lambda_explore = lambda_explore
        self.num_experts = num_experts

    def compute_balance_loss(self, routing_weights: Tensor) -> Tensor:
        expert_load = routing_weights.mean(dim=0)  # [num_experts]

        uniform = torch.ones_like(expert_load) / self.num_experts

        balance_loss = F.mse_loss(expert_load, uniform)

        return balance_loss

    def compute_exploration_loss(self, routing_weights: Tensor) -> Tensor:
        entropy = -(routing_weights * torch.log(routing_weights + 1e-10)).sum(dim=-1)

        return -entropy.mean()

    def forward(
            self,
            task_loss: Tensor,
            routing_weights: Tensor,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        balance_loss = self.compute_balance_loss(routing_weights)
        explore_loss = self.compute_exploration_loss(routing_weights)

        total_loss = (
                task_loss +
                self.lambda_balance * balance_loss +
                self.lambda_explore * explore_loss
        )

        loss_dict = {
            "task_loss": task_loss,
            "balance_loss": balance_loss,
            "explore_loss": explore_loss,
        }

        return total_loss, loss_dict


class MetaLearnerLoss(nn.Module):
    def __init__(self, lambda_reg: float = 1e-4):
        """
        Args:
            lambda_reg:
        """
        super().__init__()
        self.lambda_reg = lambda_reg
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(
            self,
            logits: Tensor,
            labels: Tensor,
            model_params: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        # BCE
        labels = labels.float().unsqueeze(-1)
        bce = self.bce_loss(logits, labels)

        # L2
        reg_loss = torch.tensor(0.0, device=logits.device)
        if model_params is not None:
            for param in model_params:
                reg_loss += torch.norm(param, p=2)
            reg_loss *= self.lambda_reg

        total_loss = bce + reg_loss

        loss_dict = {
            "bce_loss": bce,
            "reg_loss": reg_loss,
        }

        return total_loss, loss_dict


class EnsembleTrainer:
    def __init__(
            self,
            router_network: nn.Module,
            meta_learner: nn.Module,
            feature_extractor: nn.Module,
            experts: List[nn.Module],
            config: EnsembleTrainingConfig,
    ):
        self.router = router_network
        self.meta_learner = meta_learner
        self.feature_extractor = feature_extractor
        self.experts = experts
        self.config = config

        self.device = torch.device(config.device)
        self.num_experts = len(experts)

        self.router.to(self.device)
        self.meta_learner.to(self.device)
        self.feature_extractor.to(self.device)

        for expert in self.experts:
            expert.to(self.device)
            expert.eval()
            for param in expert.parameters():
                param.requires_grad = False

        self.router_loss_fn = RouterLoss(
            lambda_balance=config.lambda_balance,
            lambda_explore=config.lambda_explore,
            num_experts=self.num_experts,
        )
        self.meta_loss_fn = MetaLearnerLoss(lambda_reg=config.lambda_reg)

        ensemble_params = list(self.router.parameters()) + list(self.meta_learner.parameters())
        self.optimizer = AdamW(
            ensemble_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

    def setup_scheduler(self, num_training_steps: int):
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps,
            eta_min=1e-6,
        )

    @torch.no_grad()
    def get_expert_outputs(
            self,
            input_ids: Tensor,
            attention_mask: Optional[Tensor] = None,
            return_stats_only: bool = True,
    ) -> List[Tensor]:
        outputs = []

        for expert in self.experts:
            expert.eval()
            expert_out = expert(input_ids, attention_mask)

            if isinstance(expert_out, dict):
                logits = expert_out["logits"]
            else:
                logits = expert_out

            if logits.dim() == 3:
                logits = logits[:, -1, :]  # [batch, vocab]

            probs = F.softmax(logits, dim=-1)

            if return_stats_only:
                mean_prob = probs.mean(dim=-1, keepdim=True)
                max_prob = probs.max(dim=-1, keepdim=True)[0]
                entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1, keepdim=True)
                stats = torch.cat([mean_prob, max_prob, entropy], dim=-1)  # [batch, 3]
                outputs.append(stats)
                del logits, probs, expert_out
            else:
                outputs.append(probs)
                del logits, expert_out

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return outputs

    def build_meta_features(
            self,
            expert_outputs: List[Tensor],
            routing_weights: Tensor,
            input_features: Tensor,
            stats_only: bool = True,
    ) -> Tensor:
        features = []

        for i, output in enumerate(expert_outputs):
            if stats_only:
                weighted_stats = routing_weights[:, i:i + 1] * output
            else:
                if output.dim() == 3:
                    output = output[:, -1, :]  # [batch, vocab]

                mean_prob = output.mean(dim=-1, keepdim=True)
                max_prob = output.max(dim=-1, keepdim=True)[0]
                entropy = -(output * torch.log(output + 1e-10)).sum(dim=-1, keepdim=True)

                weighted_stats = routing_weights[:, i:i + 1] * torch.cat([
                    mean_prob, max_prob, entropy
                ], dim=-1)
            features.append(weighted_stats)

        features.append(routing_weights)

        if input_features.shape[-1] > 64:
            input_features = input_features[:, :64]
        features.append(input_features)

        meta_features = torch.cat(features, dim=-1)

        return meta_features

    def train_step(
            self,
            batch: Dict[str, Tensor],
    ) -> Dict[str, float]:

        self.router.train()
        self.meta_learner.train()

        input_ids = batch["input_ids"]
        if isinstance(input_ids, list):
            input_ids = torch.stack(input_ids) if isinstance(input_ids[0], Tensor) else torch.tensor(input_ids)
        input_ids = input_ids.to(self.device)

        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            if isinstance(attention_mask, list):
                attention_mask = torch.stack(attention_mask) if isinstance(attention_mask[0], Tensor) else torch.tensor(
                    attention_mask)
            attention_mask = attention_mask.to(self.device)

        labels = batch.get("label") or batch.get("labels")
        if isinstance(labels, list):
            labels = torch.tensor(labels, dtype=torch.long)
        elif not isinstance(labels, Tensor):
            labels = torch.tensor(labels, dtype=torch.long)
        labels = labels.to(self.device)

        with torch.no_grad():
            input_features = self.feature_extractor(input_ids, attention_mask)

        routing_weights, top_k_indices, top_k_weights = self.router(input_features)

        expert_outputs = self.get_expert_outputs(input_ids, attention_mask, return_stats_only=True)

        meta_features = self.build_meta_features(
            expert_outputs, routing_weights, input_features, stats_only=True
        )

        del expert_outputs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        meta_logits = self.meta_learner(meta_features)

        meta_params = list(self.meta_learner.parameters())
        meta_loss, meta_loss_dict = self.meta_loss_fn(meta_logits, labels, meta_params)

        router_loss, router_loss_dict = self.router_loss_fn(
            meta_loss_dict["bce_loss"], routing_weights
        )

        total_loss = meta_loss + router_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.router.parameters()) + list(self.meta_learner.parameters()),
            1.0
        )
        self.optimizer.step()

        if hasattr(self, 'scheduler'):
            self.scheduler.step()

        self.global_step += 1

        with torch.no_grad():
            preds = (torch.sigmoid(meta_logits) >= 0.5).long().squeeze()
            accuracy = (preds == labels).float().mean()

        metrics = {
            "total_loss": total_loss.item(),
            "meta_loss": meta_loss.item(),
            "router_loss": router_loss.item(),
            "bce_loss": meta_loss_dict["bce_loss"].item(),
            "balance_loss": router_loss_dict["balance_loss"].item(),
            "accuracy": accuracy.item(),
        }

        return metrics

    def train_epoch(
            self,
            dataloader: DataLoader,
            epoch: int,
    ) -> Dict[str, float]:
        self.epoch = epoch
        epoch_metrics = {
            "total_loss": 0.0,
            "meta_loss": 0.0,
            "router_loss": 0.0,
            "accuracy": 0.0,
        }
        num_steps = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Ensemble - Epoch {epoch}",
            leave=False,
        )

        for batch in progress_bar:
            step_metrics = self.train_step(batch)

            for key in epoch_metrics:
                epoch_metrics[key] += step_metrics.get(key, 0)
            num_steps += 1

            progress_bar.set_postfix({
                "loss": f"{step_metrics['total_loss']:.4f}",
                "acc": f"{step_metrics['accuracy']:.4f}",
            })

        for key in epoch_metrics:
            epoch_metrics[key] /= num_steps

        return epoch_metrics

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        self.router.eval()
        self.meta_learner.eval()

        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"]
            if isinstance(input_ids, list):
                input_ids = torch.stack(input_ids) if isinstance(input_ids[0], Tensor) else torch.tensor(input_ids)
            input_ids = input_ids.to(self.device)

            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                if isinstance(attention_mask, list):
                    attention_mask = torch.stack(attention_mask) if isinstance(attention_mask[0],
                                                                               Tensor) else torch.tensor(attention_mask)
                attention_mask = attention_mask.to(self.device)

            labels = batch.get("label") or batch.get("labels")
            if isinstance(labels, list):
                labels = torch.tensor(labels, dtype=torch.long)
            elif not isinstance(labels, Tensor):
                labels = torch.tensor(labels, dtype=torch.long)
            labels = labels.to(self.device)

            input_features = self.feature_extractor(input_ids, attention_mask)

            routing_weights, _, _ = self.router(input_features)

            expert_outputs = self.get_expert_outputs(input_ids, attention_mask, return_stats_only=True)

            meta_features = self.build_meta_features(
                expert_outputs, routing_weights, input_features, stats_only=True
            )

            del expert_outputs

            meta_logits = self.meta_learner(meta_features)
            probs = torch.sigmoid(meta_logits)
            preds = (probs >= 0.5).long().squeeze()

            loss = F.binary_cross_entropy_with_logits(
                meta_logits, labels.float().unsqueeze(-1)
            )

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += loss.item()
            num_batches += 1

            del input_ids, attention_mask, labels, input_features, routing_weights
            del meta_features, meta_logits, probs, preds, loss
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_preds = torch.tensor(all_preds)
        all_labels = torch.tensor(all_labels)

        accuracy = (all_preds == all_labels).float().mean().item()

        return {
            "loss": total_loss / num_batches,
            "accuracy": accuracy,
        }

    def train(
            self,
            train_dataloader: DataLoader,
            val_dataloader: Optional[DataLoader] = None,
            num_epochs: Optional[int] = None,
            save_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        num_epochs = num_epochs or self.config.num_epochs
        num_training_steps = len(train_dataloader) * num_epochs

        self.setup_scheduler(num_training_steps)

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
        }

        print(f"\nStarting Ensemble Training")
        print(f"  Epochs: {num_epochs}")
        print(f"  Steps per epoch: {len(train_dataloader)}")

        for epoch in range(num_epochs):
            train_metrics = self.train_epoch(train_dataloader, epoch)
            history["train_loss"].append(train_metrics["total_loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])

            if val_dataloader is not None:
                val_metrics = self.evaluate(val_dataloader)
                history["val_loss"].append(val_metrics["loss"])
                history["val_accuracy"].append(val_metrics["accuracy"])

                print(f"Epoch {epoch}: "
                      f"Train Loss={train_metrics['total_loss']:.4f}, "
                      f"Train Acc={train_metrics['accuracy']:.4f}, "
                      f"Val Loss={val_metrics['loss']:.4f}, "
                      f"Val Acc={val_metrics['accuracy']:.4f}")
            else:
                print(f"Epoch {epoch}: "
                      f"Train Loss={train_metrics['total_loss']:.4f}, "
                      f"Train Acc={train_metrics['accuracy']:.4f}")

            if save_dir and (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(save_dir, epoch)

            if train_metrics["total_loss"] < self.best_loss:
                self.best_loss = train_metrics["total_loss"]
                if save_dir:
                    self.save_checkpoint(save_dir, "best")

        return history

    def save_checkpoint(self, save_dir: str, tag):
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "router_state_dict": self.router.state_dict(),
            "meta_learner_state_dict": self.meta_learner.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
        }

        path = os.path.join(save_dir, f"ensemble_{tag}.pt")
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        self.router.load_state_dict(checkpoint["router_state_dict"])
        self.meta_learner.load_state_dict(checkpoint["meta_learner_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]


if __name__ == "__main__":
    print("Testing Ensemble Training Module...")

    from src.models import ExpertModelSmall, RouterNetwork, MetaLearner, SimpleFeatureExtractor

    num_experts = 4
    vocab_size = 1000
    hidden_dim = 256

    experts = [
        ExpertModelSmall(i, vocab_size, hidden_dim, num_layers=2)
        for i in range(num_experts)
    ]

    router = RouterNetwork(
        input_dim=768,
        num_experts=num_experts,
        hidden_dims=[256],
        top_k=2,
    )

    meta_learner = MetaLearner(
        input_dim=num_experts * 3 + num_experts + 64,  # 简化特征维度
        hidden_dims=[64, 32],
    )

    feature_extractor = SimpleFeatureExtractor(
        vocab_size=vocab_size,
        output_dim=768,
    )

    config = EnsembleTrainingConfig(
        num_epochs=2,
        batch_size=4,
        learning_rate=1e-3,
        device="cpu",
    )

    trainer = EnsembleTrainer(
        router_network=router,
        meta_learner=meta_learner,
        feature_extractor=feature_extractor,
        experts=experts,
        config=config,
    )

    class DummyMembershipDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 50

        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, vocab_size, (64,)),
                "attention_mask": torch.ones(64),
                "label": torch.randint(0, 2, ()).item(),
            }


    dataloader = DataLoader(DummyMembershipDataset(), batch_size=4)

    batch = next(iter(dataloader))
    metrics = trainer.train_step(batch)
    print(f"Step metrics: {metrics}")

    eval_metrics = trainer.evaluate(dataloader)
    print(f"Eval metrics: {eval_metrics}")

    print("\nAll tests passed!")