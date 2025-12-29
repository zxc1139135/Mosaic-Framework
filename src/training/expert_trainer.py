"""
Expert Training Module
======================

专家模型训练模块,实现论文阶段II的领域专业化训练。

论文参考:
    - 阶段II: 领域专业化训练 (Section 3)
    - 多教师知识蒸馏训练流程 (Section 3.2)
    - 损失函数定义 (公式3)
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
    """专家训练配置"""
    num_epochs: int = 160
    batch_size: int = 32
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0

    # 损失函数权重 (论文公式3)
    alpha: float = 0.6  # 领域专业化损失权重
    beta: float = 0.3  # 目标模型蒸馏损失权重
    gamma: float = 0.1  # 交叉专家蒸馏损失权重
    delta: float = 0.1  # 多样性促进损失权重

    # 蒸馏参数
    temperature: float = 2.0
    lambda1: float = 0.5  # 均值对齐权重
    lambda2: float = 0.5  # 方差对齐权重

    # 设备配置
    device: str = "cuda"
    mixed_precision: bool = True

    # GPU分配策略
    use_single_gpu: bool = False  # True: 所有专家在同一GPU, False: 分散到多个GPU
    single_gpu_id: int = 0  # 使用单GPU模式时指定的GPU ID

    # 日志配置
    log_interval: int = 100
    save_interval: int = 10


class DistillationLoss(nn.Module):
    """
    知识蒸馏损失

    包含软标签KL散度和特征对齐

    内存优化版本：只使用采样位置计算 KL 散度
    """

    def __init__(
            self,
            temperature: float = 2.0,
            lambda1: float = 0.5,
            lambda2: float = 0.5,
            sample_size: int = 32,
    ):
        """
        Args:
            temperature: 蒸馏温度
            lambda1: 均值对齐权重
            lambda2: 方差对齐权重
            sample_size: 采样的 token 位置数量（用于减少内存）
        """
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
        """
        计算蒸馏损失

        论文公式: L_target,k = KL(P_T || P_k) + λ1|μ_T - μ_k|² + λ2|σ²_T - σ²_k|²

        Args:
            student_logits: 学生模型logits [batch, seq, vocab]
            teacher_logits: 教师模型logits [batch, seq, vocab]
            student_hidden: 学生模型隐藏状态
            teacher_hidden: 教师模型隐藏状态

        Returns:
            loss: 蒸馏损失
        """
        batch_size, seq_len, vocab_size = student_logits.shape
        device = student_logits.device

        # 采样策略：只使用部分位置来计算 KL 散度
        num_samples = min(self.sample_size, seq_len)
        if seq_len > num_samples:
            indices = torch.linspace(0, seq_len - 1, num_samples).long().to(device)
        else:
            indices = torch.arange(seq_len).to(device)

        # 采样 logits
        student_logits_sampled = student_logits[:, indices, :]
        teacher_logits_sampled = teacher_logits[:, indices, :]

        # KL散度损失 (软标签)
        student_log_probs = F.log_softmax(student_logits_sampled / self.temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits_sampled / self.temperature, dim=-1)

        kl_loss = F.kl_div(
            student_log_probs,
            teacher_probs,
            reduction='batchmean',
            log_target=False
        ) * (self.temperature ** 2)

        # 清理中间变量
        del student_logits_sampled, teacher_logits_sampled, student_log_probs, teacher_probs

        # 特征对齐损失（这个操作内存消耗较小，不需要采样）
        feature_loss = torch.tensor(0.0, device=device)

        if student_hidden is not None and teacher_hidden is not None:
            # 均值对齐
            student_mean = student_hidden.mean(dim=(0, 1))
            teacher_mean = teacher_hidden.mean(dim=(0, 1))
            mean_loss = F.mse_loss(student_mean, teacher_mean)

            # 方差对齐
            student_var = student_hidden.var(dim=(0, 1))
            teacher_var = teacher_hidden.var(dim=(0, 1))
            var_loss = F.mse_loss(student_var, teacher_var)

            feature_loss = self.lambda1 * mean_loss + self.lambda2 * var_loss

        return kl_loss + feature_loss


class DiversityLoss(nn.Module):
    """
    多样性促进损失

    鼓励不同专家产生不同的输出分布

    内存优化版本：只使用采样位置计算，避免处理完整的 logits 张量
    """

    def __init__(self, num_experts: int = 8, sample_size: int = 32):
        """
        Args:
            num_experts: 专家数量
            sample_size: 采样的 token 位置数量（用于减少内存）
        """
        super().__init__()
        self.num_experts = num_experts
        self.sample_size = sample_size

    def forward(
            self,
            expert_logits: Tensor,
            other_expert_logits: List[Tensor],
    ) -> Tensor:
        """
        计算多样性损失

        论文公式: L_diversity,k = -E[Σ_{j≠k} log D(P_k, j)]

        内存优化：只采样部分位置计算 JS 散度

        Args:
            expert_logits: 当前专家的logits [batch, seq, vocab]
            other_expert_logits: 其他专家的logits列表

        Returns:
            loss: 多样性损失
        """
        if len(other_expert_logits) == 0:
            return torch.tensor(0.0, device=expert_logits.device)

        batch_size, seq_len, vocab_size = expert_logits.shape
        device = expert_logits.device

        # 采样策略：只使用部分位置来计算多样性，大幅减少内存使用
        # 采样位置：均匀采样 + 最后几个位置（通常更重要）
        num_samples = min(self.sample_size, seq_len)
        if seq_len > num_samples:
            # 均匀采样一些位置
            indices = torch.linspace(0, seq_len - 1, num_samples).long().to(device)
        else:
            indices = torch.arange(seq_len).to(device)

        # 只取采样位置的 logits [batch, num_samples, vocab]
        expert_logits_sampled = expert_logits[:, indices, :]

        # 使用 log_softmax 更数值稳定
        expert_log_probs = F.log_softmax(expert_logits_sampled, dim=-1)

        diversity_scores = []
        for other_logits in other_expert_logits:
            # 采样其他专家的 logits
            other_logits_sampled = other_logits[:, indices, :]
            other_log_probs = F.log_softmax(other_logits_sampled, dim=-1)

            # 使用简化的 JS 散度近似：基于 KL 散度的对称版本
            # 这比完整的 JS 散度更省内存
            with torch.no_grad():
                # 计算概率（只用于计算 m）
                expert_probs = expert_log_probs.exp()
                other_probs = other_log_probs.exp()
                m = 0.5 * (expert_probs + other_probs)
                m_log = (m + 1e-10).log()  # 添加小值防止 log(0)

            # 计算 JS 散度
            # JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
            kl_p_m = F.kl_div(m_log, expert_probs, reduction='batchmean', log_target=False)
            kl_q_m = F.kl_div(m_log, other_probs, reduction='batchmean', log_target=False)
            js_div = 0.5 * (kl_p_m + kl_q_m)

            diversity_scores.append(js_div)

            # 及时清理中间变量
            del other_logits_sampled, other_log_probs, other_probs, m, m_log

        # 清理
        del expert_logits_sampled, expert_log_probs, expert_probs

        # 我们希望最大化多样性，所以返回负的平均多样性
        return -torch.stack(diversity_scores).mean()


class CrossExpertDistillationLoss(nn.Module):
    """
    交叉专家蒸馏损失

    允许专家之间的知识共享

    内存优化版本：只使用采样位置计算
    """

    def __init__(self, temperature: float = 2.0, sample_size: int = 32):
        """
        Args:
            temperature: 蒸馏温度
            sample_size: 采样的 token 位置数量（用于减少内存）
        """
        super().__init__()
        self.temperature = temperature
        self.sample_size = sample_size

    def forward(
            self,
            student_logits: Tensor,
            neighbor_logits: List[Tensor],
            neighbor_weights: List[float],
    ) -> Tensor:
        """
        计算交叉专家蒸馏损失

        论文公式: L_cross,k = Σ_{j∈N_k} w_{k←j} * KL(P_j || P_k)

        Args:
            student_logits: 学生专家logits [batch, seq, vocab]
            neighbor_logits: 邻居专家logits列表
            neighbor_weights: 邻居权重列表

        Returns:
            loss: 交叉蒸馏损失
        """
        if len(neighbor_logits) == 0:
            return torch.tensor(0.0, device=student_logits.device)

        batch_size, seq_len, vocab_size = student_logits.shape
        device = student_logits.device

        # 采样策略：只使用部分位置来计算
        num_samples = min(self.sample_size, seq_len)
        if seq_len > num_samples:
            indices = torch.linspace(0, seq_len - 1, num_samples).long().to(device)
        else:
            indices = torch.arange(seq_len).to(device)

        # 采样学生 logits
        student_logits_sampled = student_logits[:, indices, :]
        student_log_probs = F.log_softmax(student_logits_sampled / self.temperature, dim=-1)

        total_loss = torch.tensor(0.0, device=device)
        for neighbor_logit, weight in zip(neighbor_logits, neighbor_weights):
            # 采样邻居 logits
            neighbor_logits_sampled = neighbor_logit[:, indices, :]
            neighbor_probs = F.softmax(neighbor_logits_sampled / self.temperature, dim=-1)

            kl_loss = F.kl_div(student_log_probs, neighbor_probs, reduction='batchmean', log_target=False)
            total_loss = total_loss + weight * kl_loss

            # 清理中间变量
            del neighbor_logits_sampled, neighbor_probs

        # 清理
        del student_logits_sampled, student_log_probs

        return total_loss * (self.temperature ** 2)


class ExpertTrainer:
    """
    专家模型训练器

    实现论文阶段II的完整训练流程
    """

    def __init__(
            self,
            expert_model: nn.Module,
            expert_id: int,
            config: ExpertTrainingConfig,
            target_model: Optional[nn.Module] = None,
            other_experts: Optional[List[nn.Module]] = None,
            device_id: Optional[int] = None,  # 新增：指定设备ID
    ):
        """
        Args:
            expert_model: 专家模型
            expert_id: 专家ID
            config: 训练配置
            target_model: 目标模型 (用于蒸馏)
            other_experts: 其他专家模型列表 (用于交叉蒸馏和多样性)
            device_id: 指定的GPU设备ID (None则使用config中的设备)
        """
        self.expert_model = expert_model
        self.expert_id = expert_id
        self.config = config
        self.target_model = target_model
        self.other_experts = other_experts or []

        # 设备分配逻辑
        if device_id is not None:
            # 使用指定的设备ID
            if config.device == "cuda" and torch.cuda.is_available():
                self.device = torch.device(f"cuda:{device_id}")
            else:
                self.device = torch.device("cpu")
        else:
            # 使用config中的设备
            self.device = torch.device(config.device)

        # 将模型移至设备
        self.expert_model.to(self.device)

        # 处理 target_model - 如果多个训练器共享，需要特别小心
        if self.target_model is not None:
            # 只在推理时临时移到正确的设备，不永久修改其位置
            # 这样避免了多个训练器同时尝试移动同一个模型
            self.target_model.eval()
            self._target_model_device = next(self.target_model.parameters()).device
            print(f"Expert {expert_id}: Target model on device {self._target_model_device}, will use on {self.device}")
        else:
            self._target_model_device = None

        # 损失函数
        self.distillation_loss = DistillationLoss(
            temperature=config.temperature,
            lambda1=config.lambda1,
            lambda2=config.lambda2,
        )
        self.diversity_loss = DiversityLoss(len(self.other_experts) + 1)
        self.cross_distill_loss = CrossExpertDistillationLoss(config.temperature)

        # 优化器
        self.optimizer = AdamW(
            self.expert_model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 混合精度训练
        self.scaler = torch.amp.GradScaler('cuda') if config.mixed_precision else None

        # 训练状态
        self.global_step = 0
        self.epoch = 0
        self.best_loss = float('inf')

    def setup_scheduler(self, num_training_steps: int):
        """设置学习率调度器"""
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

        # 组合调度器
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
        """
        计算总损失

        论文公式: L_k = α·L_domain + β·L_target + γ·L_cross + δ·L_diversity

        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
            labels: 标签 (可选)

        Returns:
            total_loss: 总损失
            loss_components: 各损失分量字典
        """
        # 前向传播
        outputs = self.expert_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        # 处理不同类型的输出格式 (字典、对象、或直接张量)
        if isinstance(outputs, dict):
            # 字典类型输出
            logits = outputs.get("logits")
            hidden_states = outputs.get("last_hidden_state")
            if hidden_states is None and outputs.get("hidden_states"):
                hidden_states = outputs["hidden_states"][-1] if isinstance(outputs["hidden_states"], (list, tuple)) else \
                outputs["hidden_states"]
        elif hasattr(outputs, 'logits'):
            # 对象类型输出 (如 HuggingFace 模型)
            logits = outputs.logits
            hidden_states = outputs.hidden_states[-1] if hasattr(outputs,
                                                                 'hidden_states') and outputs.hidden_states else None
        else:
            # 直接张量输出
            logits = outputs
            hidden_states = None

        # 1. 领域专业化损失 (语言模型损失)
        if labels is not None:
            domain_loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
        else:
            # 使用自回归预测作为监督
            domain_loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                input_ids[:, 1:].contiguous().view(-1),
                ignore_index=-100,
            )

        # 2. 目标模型蒸馏损失
        target_loss = torch.tensor(0.0, device=self.device)
        if self.target_model is not None:
            with torch.no_grad():
                # 确定 target_model 所在的设备
                target_device = next(self.target_model.parameters()).device

                # 只在设备不同时才进行数据转移
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

                # 处理不同类型的输出格式
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

                # 将输出移回当前设备
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

            # 清理 target_model 的中间变量
            del target_outputs, target_logits, target_hidden
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 3. 交叉专家蒸馏损失和4. 多样性促进损失
        # 合并计算以避免重复前向传播
        cross_loss = torch.tensor(0.0, device=self.device)
        diversity_loss = torch.tensor(0.0, device=self.device)

        if len(self.other_experts) > 0:
            neighbor_logits = []
            neighbor_weights = []

            with torch.no_grad():
                for other_expert in self.other_experts:
                    # 获取其他专家所在的设备
                    expert_device = next(other_expert.parameters()).device

                    # 只在设备不同时才进行数据转移（单GPU模式下会跳过）
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

                    # 处理不同类型的输出格式
                    if isinstance(other_outputs, dict):
                        other_logits = other_outputs.get("logits")
                    elif hasattr(other_outputs, 'logits'):
                        other_logits = other_outputs.logits
                    else:
                        other_logits = other_outputs

                    # 将输出移回当前设备并 detach
                    if expert_device != self.device:
                        other_logits = other_logits.to(self.device).detach()
                    else:
                        other_logits = other_logits.detach()

                    neighbor_logits.append(other_logits)
                    neighbor_weights.append(1.0 / len(self.other_experts))

                    # 清理其他输出
                    del other_outputs

            # 计算交叉专家蒸馏损失
            cross_loss = self.cross_distill_loss(
                student_logits=logits,
                neighbor_logits=neighbor_logits,
                neighbor_weights=neighbor_weights,
            )

            # 计算多样性促进损失 (使用相同的 neighbor_logits)
            diversity_loss = self.diversity_loss(logits, neighbor_logits)

            # 清理 neighbor_logits 释放内存
            del neighbor_logits, neighbor_weights
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # 总损失
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
        """
        单步训练

        Args:
            batch: 批次数据

        Returns:
            metrics: 步骤指标
        """
        self.expert_model.train()

        # 将数据移至设备
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch.get("labels")
        if labels is not None:
            labels = labels.to(self.device)

        # 前向传播与损失计算
        if self.config.mixed_precision and self.scaler is not None:
            with torch.amp.autocast('cuda'):
                total_loss, loss_components = self.compute_loss(
                    input_ids, attention_mask, labels
                )

            # 反向传播 (缩放梯度)
            self.scaler.scale(total_loss).backward()
        else:
            total_loss, loss_components = self.compute_loss(
                input_ids, attention_mask, labels
            )

            # 反向传播
            total_loss.backward()

        # 返回指标
        metrics = {
            "total_loss": total_loss.item(),
            "domain_loss": loss_components["domain_loss"].item(),
            "target_loss": loss_components["target_loss"].item(),
            "cross_loss": loss_components["cross_loss"].item(),
            "diversity_loss": loss_components["diversity_loss"].item(),
        }

        return metrics

    def optimizer_step(self):
        """优化器步骤"""
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
        """
        训练一个epoch

        Args:
            dataloader: 数据加载器
            epoch: 当前epoch

        Returns:
            metrics: epoch平均指标
        """
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

            # 累积指标
            for key in epoch_metrics:
                epoch_metrics[key] += step_metrics[key]
            num_steps += 1

            # 梯度累积后更新
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer_step()

            # 更新进度条
            if step % self.config.log_interval == 0:
                progress_bar.set_postfix({
                    "loss": f"{step_metrics['total_loss']:.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })

        # 计算平均值
        for key in epoch_metrics:
            epoch_metrics[key] /= num_steps

        return epoch_metrics

    def train(
            self,
            dataloader: DataLoader,
            num_epochs: Optional[int] = None,
            save_dir: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        完整训练流程

        Args:
            dataloader: 数据加载器
            num_epochs: 训练轮数
            save_dir: 保存目录

        Returns:
            history: 训练历史
        """
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

            # 记录历史
            for key in history:
                history[key].append(epoch_metrics[key])

            # 打印epoch摘要
            print(f"Expert {self.expert_id} - Epoch {epoch}: "
                  f"Loss={epoch_metrics['total_loss']:.4f}, "
                  f"Domain={epoch_metrics['domain_loss']:.4f}, "
                  f"Target={epoch_metrics['target_loss']:.4f}")

            # 保存检查点
            if save_dir and (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(save_dir, epoch)

            # 更新最佳损失
            if epoch_metrics['total_loss'] < self.best_loss:
                self.best_loss = epoch_metrics['total_loss']
                if save_dir:
                    self.save_checkpoint(save_dir, "best")

        return history

    def save_checkpoint(self, save_dir: str, tag: str):
        """保存检查点"""
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
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)

        self.expert_model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_loss = checkpoint["best_loss"]


class MultiExpertTrainer:
    """
    多专家并行训练器

    协调所有专家的训练过程，智能分配GPU设备
    """

    def __init__(
            self,
            experts: List[nn.Module],
            config: ExpertTrainingConfig,
            target_model: Optional[nn.Module] = None,
    ):
        """
        Args:
            experts: 专家模型列表
            config: 训练配置
            target_model: 目标模型
        """
        self.experts = experts
        self.config = config
        self.target_model = target_model
        self.num_experts = len(experts)

        # 重要：先将所有模型移到CPU，避免之前的错误设备分配
        print("Moving all models to CPU first to reset device state...")
        for i, expert in enumerate(self.experts):
            expert.cpu()
        if self.target_model is not None:
            self.target_model.cpu()

        # 清空CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # 获取可用的GPU数量
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

        # 根据配置决定设备分配策略
        if num_gpus == 0:
            print("Warning: No GPU available, using CPU")
            device_ids = [None] * self.num_experts  # 全部使用CPU
        elif config.use_single_gpu:
            # 单GPU模式：所有专家使用同一个GPU
            gpu_id = min(config.single_gpu_id, num_gpus - 1)  # 确保GPU ID有效
            device_ids = [gpu_id] * self.num_experts
            print(f"Single GPU mode: All {self.num_experts} experts on GPU {gpu_id}")
        else:
            # 多GPU模式：专家均匀分配到可用的GPU上
            print(f"Multi-GPU mode: Available GPUs: {num_gpus}")
            device_ids = [i % num_gpus for i in range(self.num_experts)]
            print(f"Device allocation for {self.num_experts} experts:")
            for expert_id, device_id in enumerate(device_ids):
                print(f"  Expert {expert_id} -> GPU {device_id}")

        # 为每个专家创建训练器
        self.trainers = []
        for i, expert in enumerate(experts):
            # 注意：不传递 other_experts，避免设备冲突
            # other_experts 会在需要时临时移到正确的设备
            trainer = ExpertTrainer(
                expert_model=expert,
                expert_id=i,
                config=config,
                target_model=target_model,
                other_experts=[],  # 先传空列表
                device_id=device_ids[i],  # 指定设备ID
            )
            self.trainers.append(trainer)

        # 现在为每个训练器设置 other_experts（已经在正确的设备上）
        for i, trainer in enumerate(self.trainers):
            other_experts = []
            for j, other_trainer in enumerate(self.trainers):
                if j != i:
                    other_experts.append(other_trainer.expert_model)
            trainer.other_experts = other_experts
            # 更新损失函数
            trainer.diversity_loss = DiversityLoss(len(other_experts) + 1)

    def train(
            self,
            domain_dataloaders: Dict[int, DataLoader],
            num_epochs: Optional[int] = None,
            save_dir: Optional[str] = None,
    ) -> Dict[int, Dict[str, List[float]]]:
        """
        训练所有专家

        Args:
            domain_dataloaders: 领域数据加载器字典
            num_epochs: 训练轮数
            save_dir: 保存目录

        Returns:
            histories: 所有专家的训练历史
        """
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
    # 测试代码
    print("Testing Expert Training Module...")

    from src.models import ExpertModelSmall

    # 创建测试模型
    expert = ExpertModelSmall(expert_id=0, vocab_size=1000, hidden_dim=256, num_layers=2)

    # 创建配置
    config = ExpertTrainingConfig(
        num_epochs=2,
        batch_size=4,
        learning_rate=1e-4,
        gradient_accumulation_steps=1,
        mixed_precision=False,
    )

    # 创建训练器
    trainer = ExpertTrainer(expert, expert_id=0, config=config)


    # 创建测试数据
    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return {
                "input_ids": torch.randint(0, 1000, (128,)),
                "attention_mask": torch.ones(128),
            }


    dataloader = DataLoader(DummyDataset(), batch_size=4)

    # 测试训练步骤
    batch = next(iter(dataloader))
    metrics = trainer.train_step(batch)
    print(f"Step metrics: {metrics}")

    # 测试优化器步骤
    trainer.optimizer_step()
    print(f"Global step: {trainer.global_step}")

    print("\nAll tests passed!")