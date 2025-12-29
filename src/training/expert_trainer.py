"""
Expert Training Module
======================
"""

import os
import time
from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import warnings

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
warnings.filterwarnings('ignore')


@dataclass
class ExpertTrainingConfig:
    num_epochs: int = 160
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    alpha: float = 0.6
    beta: float = 0.3
    gamma: float = 0.1
    delta: float = 0.1

    temperature: float = 2.0
    lambda1: float = 0.5
    lambda2: float = 0.5

    device: str = "cuda"
    mixed_precision: bool = True

    use_single_gpu: bool = False
    single_gpu_id: int = 0

    log_interval: int = 100
    save_interval: int = 10


class DistillationLoss(nn.Module):
    def __init__(
            self,
            temperature: float = 2.0,
            lambda1: float = 0.5,
            lambda2: float = 0.5,
            sample_size: int = 32,
    ):
        super().__init__()
        self.temperature = temperature
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.sample_size = sample_size

    def forward(
            self,
            student_logits: Tensor,
            teacher_logits: Tensor,
            student_hidden: Optional[Tensor] = None,
            teacher_hidden: Optional[Tensor] = None,
    ) -> Tensor:
        batch_size, seq_len, vocab_size = student_logits.shape
        device = student_logits.device

        num_samples = min(self.sample_size, seq_len)
        if seq_len > num_samples:
            indices = torch.linspace(0, seq_len - 1, num_samples).long().to(device)
        else:
            indices = torch.arange(seq_len).to(device)

        student_logits_sampled = student_logits[:, indices, :]
        teacher_logits_sampled = teacher_logits[:, indices, :]

        student_log_probs = F.log_softmax(student_logits_sampled / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits_sampled / self.temperature, dim=-1)

        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean',
            log_target=False
        ) * (self.temperature ** 2)

        del student_logits_sampled, teacher_logits_sampled, student_log_probs, teacher_probs

        feature_loss = torch.tensor(0.0, device=device)

        if student_hidden is not None and teacher_hidden is not None:
            student_mean = student_hidden.mean(dim=(0, 1))
            teacher_mean = teacher_hidden.mean(dim=(0, 1))
            mean_loss = F.mse_loss(student_mean, teacher_mean)

            student_var = student_hidden.var(dim=(0, 1))
            teacher_var = teacher_hidden.var(dim=(0, 1))
            var_loss = F.mse_loss(student_var, teacher_var)

            feature_loss = self.lambda1 * mean_loss + self.lambda2 * var_loss

        return kl_loss + feature_loss


class DiversityLoss(nn.Module):
    def __init__(self, num_experts: int = 8, sample_size: int = 32):
        super().__init__()
        self.num_experts = num_experts
        self.sample_size = sample_size

    def forward(
            self,
            expert_logits: Tensor,
            other_expert_logits: List[Tensor],
    ) -> Tensor:
        if len(other_expert_logits) == 0:
            return torch.tensor(0.0, device=expert_logits.device)

        batch_size, seq_len, vocab_size = expert_logits.shape
        device = expert_logits.device

        num_samples = min(self.sample_size, seq_len)
        if seq_len > num_samples:
            indices = torch.linspace(0, seq_len - 1, num_samples).long().to(device)
        else:
            indices = torch.arange(seq_len).to(device)

        # logits [batch, num_samples, vocab]
        expert_logits_sampled = expert_logits[:, indices, :]

        expert_log_probs = F.log_softmax(expert_logits_sampled, dim=-1)

        diversity_scores = []
        for other_logits in other_expert_logits:
            other_logits_sampled = other_logits[:, indices, :]
            other_log_probs = F.log_softmax(other_logits_sampled, dim=-1)

            with torch.no_grad():
                expert_probs = expert_log_probs.exp()
                other_probs = other_log_probs.exp()
                m = 0.5 * (expert_probs + other_probs)
                m_log = (m + 1e-10).log()

            # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
            kl_p_m = F.kl_div(m_log, expert_probs, reduction='batchmean', log_target=False)
            kl_q_m = F.kl_div(m_log, other_probs, reduction='batchmean', log_target=False)
            js_div = 0.5 * (kl_p_m + kl_q_m)

            diversity_scores.append(js_div)

            del other_logits_sampled, other_log_probs, other_probs, m, m_log

        del expert_logits_sampled, expert_log_probs, expert_probs

        return -torch.stack(diversity_scores).mean()


class CrossExpertDistillationLoss(nn.Module):
    def __init__(self, temperature: float = 2.0, sample_size: int = 32):
        super().__init__()
        self.temperature = temperature
        self.sample_size = sample_size

    def forward(
            self,
            student_logits: Tensor,
            neighbor_logits: List[Tensor],
            neighbor_weights: List[float],
    ) -> Tensor:
        if len(neighbor_logits) == 0:
            return torch.tensor(0.0, device=student_logits.device)

        batch_size, seq_len, vocab_size = student_logits.shape
        device = student_logits.device

        num_samples = min(self.sample_size, seq_len)
        if seq_len > num_samples:
            indices = torch.linspace(0, seq_len - 1, num_samples).long().to(device)
        else:
            indices = torch.arange(seq_len).to(device)

        student_logits_sampled = student_logits[:, indices, :]
        student_log_probs = F.log_softmax(student_logits_sampled / self.temperature, dim=-1)

        total_loss = torch.tensor(0.0, device=device)
        for neighbor_logit, weight in zip(neighbor_logits, neighbor_weights):
            neighbor_logits_sampled = neighbor_logit[:, indices, :]
            neighbor_probs = F.softmax(neighbor_logits_sampled / self.temperature, dim=-1)

            kl_loss = F.kl_div(student_log_probs, neighbor_probs, reduction='batchmean', log_target=False)
            total_loss = total_loss + weight * kl_loss

            del neighbor_logits_sampled, neighbor_probs

        del student_logits_sampled, student_log_probs

        return total_loss * (self.temperature ** 2)


class ExpertTrainer:
    def __init__(
            self,
            expert_model: nn.Module,
            expert_id: int,
            config: ExpertTrainingConfig,
            target_model: Optional[nn.Module] = None,
            other_experts: Optional[List[nn.Module]] = None,
            device_id: Optional[int] = None,  # 新增：指定设备ID
    ):
        self.expert_model = expert_model
        self.expert_id = expert_id
        self.config = config
        self.target_model = target_model
        self.other_experts = other_experts or []

        if device_id is not None:
            if config.device == "cuda" and torch.cuda.is_available():
                self.device = torch.device(f"cuda:{device_id}")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(config.device)

        self.expert_model.to(self.device)

        if self.target_model is not None:
            self.target_model.eval()
            self._target_model_device = next(self.target_model.parameters()).device
            print(f"Expert {expert_id}: Target model on device {self._target_model_device}, will use on {self.device}")
        else:
            self._target_model_device = None

        self.distillation_loss = DistillationLoss(
            temperature=config.temperature,
            lambda1=config.lambda1,
            lambda2=config.lambda2,
        )
        self.diversity_loss = DiversityLoss(len(self.other_experts) + 1)
        self.cross_distill_loss = CrossExpertDistillationLoss(config.temperature)

        self.optimizer = AdamW(
            self.expert_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None

        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

    def setup_scheduler(self, num_training_steps: int):
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.config.warmup_steps,
        )

        # Cosine annealing scheduler
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_training_steps - self.config.warmup_steps,
            eta_min=self.config.learning_rate * 0.1,
        )

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_steps],
        )

    def compute_loss(
            self,
            input_ids: Tensor,
            attention_mask: Tensor,
            labels: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        outputs = self.expert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        if isinstance(outputs, dict):
            logits = outputs.get("logits")
            hidden_states = outputs.get("last_hidden_state")
            if hidden_states is None and outputs.get("hidden_states"):
                hidden_states = outputs["hidden_states"][-1] if isinstance(outputs["hidden_states"], (list, tuple)) else \
                outputs["hidden_states"]
        elif hasattr(outputs, 'logits'):
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs,
                                                                 'hidden_states') and outputs.hidden_states else None
        else:
            logits = outputs
            hidden_states = None

        if labels is not None:
            domain_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        else:
            domain_loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                input_ids[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

        target_loss = torch.tensor(0.0, device=self.device)
        if self.target_model is not None:
            with torch.no_grad():
                target_device = next(self.target_model.parameters()).device

                if target_device != self.device:
                    target_input_ids = input_ids.to(target_device)
                    target_attention_mask = attention_mask.to(target_device)
                else:
                    target_input_ids = input_ids
                    target_attention_mask = attention_mask

                target_outputs = self.target_model(
                    input_ids=target_input_ids,
                    attention_mask=target_attention_mask,
                    output_hidden_states=True,
                )

                if isinstance(target_outputs, dict):
                    target_logits = target_outputs.get("logits")
                    target_hidden = target_outputs.get("last_hidden_state")
                    if target_hidden is None and target_outputs.get("hidden_states"):
                        target_hidden = target_outputs["hidden_states"][-1] if isinstance(
                            target_outputs["hidden_states"], (list, tuple)) else target_outputs["hidden_states"]
                elif hasattr(target_outputs, 'logits'):
                    target_logits = target_outputs.logits
                    target_hidden = target_outputs.hidden_states[-1] if hasattr(target_outputs,
                                                                                'hidden_states') and target_outputs.hidden_states else None
                else:
                    target_logits = target_outputs
                    target_hidden = None

                if target_device != self.device:
                    target_logits = target_logits.to(self.device)
                    if target_hidden is not None:
                        target_hidden = target_hidden.to(self.device)

            target_loss = self.distillation_loss(
                student_logits=logits,
                teacher_logits=target_logits,
                student_hidden=hidden_states,
                teacher_hidden=target_hidden,
            )

            del target_outputs, target_logits, target_hidden
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        cross_loss = torch.tensor(0.0, device=self.device)
        diversity_loss = torch.tensor(0.0, device=self.device)

        if len(self.other_experts) > 0:
            neighbor_logits = []
            neighbor_weights = []

            with torch.no_grad():
                for other_expert in self.other_experts:
                    expert_device = next(other_expert.parameters()).device

                    if expert_device != self.device:
                        expert_input_ids = input_ids.to(expert_device)
                        expert_attention_mask = attention_mask.to(expert_device)
                    else:
                        expert_input_ids = input_ids
                        expert_attention_mask = attention_mask

                    other_outputs = other_expert(
                        input_ids=expert_input_ids,
                        attention_mask=expert_attention_mask,
                    )

                    if isinstance(other_outputs, dict):
                        other_logits = other_outputs.get("logits")
                    elif hasattr(other_outputs, 'logits'):
                        other_logits = other_outputs.logits
                    else:
                        other_logits = other_outputs

                    if expert_device != self.device:
                        other_logits = other_logits.to(self.device).detach()
                    else:
                        other_logits = other_logits.detach()

                    neighbor_logits.append(other_logits)
                    neighbor_weights.append(1.0 / len(self.other_experts))

                    del other_outputs

            cross_loss = self.cross_distill_loss(
                student_logits=logits,
                neighbor_logits=neighbor_logits,
                neighbor_weights=neighbor_weights,
            )

            diversity_loss = self.diversity_loss(logits, neighbor_logits)

            del neighbor_logits, neighbor_weights
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        total_loss = (
                self.config.alpha * domain_loss +
                self.config.beta * target_loss +
                self.config.gamma * cross_loss +
                self.config.delta * diversity_loss
        )

        loss_components = {
            "domain_loss": domain_loss,
            "target_loss": target_loss,
            "cross_loss": cross_loss,
            "diversity_loss": diversity_loss,
        }

        return total_loss, loss_components

    def train_step(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        self.expert_model.train()

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(self.device)

        if self.config.mixed_precision and self.scaler is not None:
            with torch.amp.autocast('cuda'):
                total_loss, loss_components = self.compute_loss(
                    input_ids, attention_mask, labels
                )

            self.scaler.scale(total_loss).backward()
        else:
            total_loss, loss_components = self.compute_loss(
                input_ids, attention_mask, labels
            )

            total_loss.backward()

        metrics = {
            "total_loss": total_loss.item(),
            "domain_loss": loss_components["domain_loss"].item(),
            "target_loss": loss_components["target_loss"].item(),
            "cross_loss": loss_components["cross_loss"].item(),
            "diversity_loss": loss_components["diversity_loss"].item(),
        }

        return metrics

    def optimizer_step(self):
        if self.config.mixed_precision and self.scaler is not None:
            torch.nn.utils.clip_grad_norm_(
                self.expert_model.parameters(),
                self.config.max_grad_norm,
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(
                self.expert_model.parameters(),
                self.config.max_grad_norm,
            )
            self.optimizer.step()

        self.optimizer.zero_grad()

        if hasattr(self, 'scheduler'):
            self.scheduler.step()

        self.global_step += 1

    def train_epoch(
            self,
            dataloader: DataLoader,
            epoch: int,
    ) -> Dict[str, float]:
        self.epoch = epoch
        epoch_metrics = {
            "total_loss": 0.0,
            "domain_loss": 0.0,
            "target_loss": 0.0,
            "cross_loss": 0.0,
            "diversity_loss": 0.0,
        }
        num_steps = 0

        progress_bar = tqdm(
            dataloader,
            desc=f"Expert {self.expert_id} - Epoch {epoch}",
            leave=False,
        )

        for step, batch in enumerate(progress_bar):
            step_metrics = self.train_step(batch)

            for key in epoch_metrics:
                epoch_metrics[key] += step_metrics[key]
            num_steps += 1

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer_step()

            if step % self.config.log_interval == 0:
                progress_bar.set_postfix({
                    "loss": f"{step_metrics['total_loss']:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })

        for key in epoch_metrics:
            epoch_metrics[key] /= num_steps

        return epoch_metrics

    def train(
            self,
            dataloader: DataLoader,
            num_epochs: Optional[int] = None,
            save_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        num_epochs = num_epochs or self.config.num_epochs
        num_training_steps = len(dataloader) * num_epochs // self.config.gradient_accumulation_steps

        self.setup_scheduler(num_training_steps)

        history = {
            "total_loss": [],
            "domain_loss": [],
            "target_loss": [],
            "cross_loss": [],
            "diversity_loss": [],
        }

        print(f"\nStarting training for Expert {self.expert_id} on {self.device}")
        print(f"  Epochs: {num_epochs}")
        print(f"  Steps per epoch: {len(dataloader)}")
        print(f"  Total steps: {num_training_steps}")

        for epoch in range(num_epochs):
            epoch_metrics = self.train_epoch(dataloader, epoch)

            for key in history:
                history[key].append(epoch_metrics[key])

            print(f"Expert {self.expert_id} - Epoch {epoch}: "
                  f"Loss={epoch_metrics['total_loss']:.4f}, "
                  f"Domain={epoch_metrics['domain_loss']:.4f}, "
                  f"Target={epoch_metrics['target_loss']:.4f}")

            if save_dir and (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(save_dir, epoch)

            if epoch_metrics['total_loss'] < self.best_loss:
                self.best_loss = epoch_metrics['total_loss']
                if save_dir:
                    self.save_checkpoint(save_dir, "best")

        return history

    def save_checkpoint(self, save_dir: str, tag: str):
        os.makedirs(save_dir, exist_ok=True)

        checkpoint = {
            "expert_id": self.expert_id,
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.expert_model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
        }

        path = os.path.join(save_dir, f"expert_{self.expert_id}_{tag}.pt")
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)

        self.expert_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]


class MultiExpertTrainer:
    def __init__(
            self,
            experts: List[nn.Module],
            config: ExpertTrainingConfig,
            target_model: Optional[nn.Module] = None,
    ):
        self.experts = experts
        self.config = config
        self.target_model = target_model
        self.num_experts = len(experts)

        print("Moving all models to CPU first to reset device state...")
        for i, expert in enumerate(self.experts):
            expert.cpu()
        if self.target_model is not None:
            self.target_model.cpu()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        if num_gpus == 0:
            print("Warning: No GPU available, using CPU")
            device_ids = [None] * self.num_experts
        elif config.use_single_gpu:
            gpu_id = min(config.single_gpu_id, num_gpus - 1)
            device_ids = [gpu_id] * self.num_experts
            print(f"Single GPU mode: All {self.num_experts} experts on GPU {gpu_id}")
        else:
            print(f"Multi-GPU mode: Available GPUs: {num_gpus}")
            device_ids = [i % num_gpus for i in range(self.num_experts)]
            print(f"Device allocation for {self.num_experts} experts:")
            for expert_id, device_id in enumerate(device_ids):
                print(f"  Expert {expert_id} -> GPU {device_id}")

        self.trainers = []
        for i, expert in enumerate(experts):
            trainer = ExpertTrainer(
                expert_model=expert,
                expert_id=i,
                config=config,
                target_model=target_model,
                other_experts=[],
                device_id=device_ids[i],
            )
            self.trainers.append(trainer)

        for i, trainer in enumerate(self.trainers):
            other_experts = []
            for j, other_trainer in enumerate(self.trainers):
                if j != i:
                    other_experts.append(other_trainer.expert_model)
            trainer.other_experts = other_experts
            trainer.diversity_loss = DiversityLoss(len(other_experts) + 1)

    def train(
            self,
            domain_dataloaders: Dict[int, DataLoader],
            num_epochs: Optional[int] = None,
            save_dir: Optional[str] = None,
    ) -> Dict[int, Dict[str, List[float]]]:
        histories = {}

        for expert_id, trainer in enumerate(self.trainers):
            if expert_id in domain_dataloaders:
                dataloader = domain_dataloaders[expert_id]
                expert_save_dir = os.path.join(save_dir, f"expert_{expert_id}") if save_dir else None

                history = trainer.train(
                    dataloader,
                    num_epochs=num_epochs,
                    save_dir=expert_save_dir,
                )
                histories[expert_id] = history
            else:
                print(f"Warning: No dataloader for expert {expert_id}")

        return histories


if __name__ == "__main__":
    print("Testing Expert Training Module...")

    from src.models import ExpertModelSmall

    expert = ExpertModelSmall(expert_id=0, vocab_size=1000, hidden_dim=256, num_layers=2)

    config = ExpertTrainingConfig(
        num_epochs=2,
        batch_size=4,
        learning_rate=1e-4,
        gradient_accumulation_steps=1,
        mixed_precision=False,
    )

    trainer = ExpertTrainer(expert, expert_id=0, config=config)


    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 1000, (128,)),
                "attention_mask": torch.ones(128),
            }


    dataloader = DataLoader(DummyDataset(), batch_size=4)

    batch = next(iter(dataloader))
    metrics = trainer.train_step(batch)
    print(f"Step metrics: {metrics}")

    trainer.optimizer_step()
    print(f"Global step: {trainer.global_step}")

    print("\nAll tests passed!")