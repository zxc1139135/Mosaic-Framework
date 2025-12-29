"""
Attack Training Module
======================

攻击分类器训练模块，实现论文阶段IV的黑盒成员推理攻击执行。

论文参考:
    - 阶段IV: 黑盒成员推理攻击执行 (Section 5)
    - 攻击分类器训练 (Section 5.2)
    - 损失函数 (公式9)
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
    """攻击训练配置"""
    num_epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # 早停配置
    early_stopping_patience: int = 5

    # 数据划分
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # 设备配置
    device: str = "cuda"

    # 日志配置
    log_interval: int = 10


class EarlyStopping:
    """
    早停机制

    当验证集损失连续patience个epoch不下降时停止训练
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        """
        Args:
            patience: 容忍的epoch数
            min_delta: 最小改善阈值
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss: float) -> bool:
        """
        检查是否应该早停

        Args:
            val_loss: 验证集损失

        Returns:
            should_stop: 是否应该停止
        """
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
    """
    攻击分类器训练器

    论文公式(9): L_attack = -1/N Σ[y_i log p(x_i) + (1-y_i) log(1-p(x_i))] + λ||θ||²
    """

    def __init__(
            self,
            attack_classifier: nn.Module,
            config: AttackTrainingConfig,
    ):
        """
        Args:
            attack_classifier: 攻击分类器模型
            config: 训练配置
        """
        self.classifier = attack_classifier
        self.config = config

        self.device = torch.device(config.device)
        self.classifier.to(self.device)

        # 损失函数
        self.criterion = nn.BCEWithLogitsLoss()

        # 优化器
        self.optimizer = Adam(
            self.classifier.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # 学习率调度器
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
        )

        # 早停
        self.early_stopping = EarlyStopping(patience=config.early_stopping_patience)

        # 训练状态
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.best_model_state = None

    def prepare_data(
            self,
            features: np.ndarray,
            labels: np.ndarray,
    ) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        准备训练、验证和测试数据

        论文5.2节: 数据划分按7:3比例 (这里细分为7:1.5:1.5)

        Args:
            features: 特征矩阵 [N, feature_dim]
            labels: 标签 [N]

        Returns:
            train_loader, val_loader, test_loader
        """
        # Z-score标准化
        mean = features.mean(axis=0)
        std = features.std(axis=0) + 1e-6
        features_normalized = (features - mean) / std

        # 保存标准化参数
        self.feature_mean = torch.tensor(mean, dtype=torch.float32)
        self.feature_std = torch.tensor(std, dtype=torch.float32)

        # 划分数据
        # 首先划分出测试集
        X_temp, X_test, y_temp, y_test = train_test_split(
            features_normalized, labels,
            test_size=self.config.test_ratio,
            random_state=42,
            stratify=labels,
        )

        # 从剩余数据中划分训练集和验证集
        val_ratio_adjusted = self.config.val_ratio / (1 - self.config.test_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio_adjusted,
            random_state=42,
            stratify=y_temp,
        )

        # 转换为Tensor
        X_train = torch.tensor(X_train, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_val = torch.tensor(X_val, dtype=torch.float32)
        y_val = torch.tensor(y_val, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # 创建DataLoader
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
        """
        单步训练

        Args:
            batch: (features, labels)

        Returns:
            metrics: 训练指标
        """
        self.classifier.train()

        features, labels = batch
        features = features.to(self.device)
        labels = labels.to(self.device).unsqueeze(-1)

        # 前向传播
        logits = self.classifier(features)

        # 计算损失 (论文公式9)
        loss = self.criterion(logits, labels)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 计算准确率
        with torch.no_grad():
            preds = (torch.sigmoid(logits) >= 0.5).float()
            accuracy = (preds == labels).float().mean()

        return {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """
        评估模型

        Args:
            dataloader: 数据加载器

        Returns:
            metrics: 评估指标
        """
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

        # 计算指标
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
        """训练一个epoch"""
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
        """
        完整训练流程

        Args:
            features: 特征矩阵
            labels: 标签
            save_dir: 保存目录

        Returns:
            history: 训练历史
        """
        # 准备数据
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

            # 训练
            train_metrics = self.train_epoch(train_loader)
            history["train_loss"].append(train_metrics["loss"])
            history["train_accuracy"].append(train_metrics["accuracy"])

            # 验证
            val_metrics = self.evaluate(val_loader)
            history["val_loss"].append(val_metrics["loss"])
            history["val_accuracy"].append(val_metrics["accuracy"])

            # 学习率调度
            self.scheduler.step(val_metrics["loss"])

            # 打印进度
            print(f"Epoch {epoch}: "
                  f"Train Loss={train_metrics['loss']:.4f}, "
                  f"Train Acc={train_metrics['accuracy']:.4f}, "
                  f"Val Loss={val_metrics['loss']:.4f}, "
                  f"Val Acc={val_metrics['accuracy']:.4f}")

            # 保存最佳模型
            if val_metrics["loss"] < self.best_val_loss:
                self.best_val_loss = val_metrics["loss"]
                self.best_model_state = self.classifier.state_dict().copy()

                if save_dir:
                    self.save_checkpoint(save_dir, "best")

            # 早停检查
            if self.early_stopping(val_metrics["loss"]):
                print(f"Early stopping at epoch {epoch}")
                break

        # 恢复最佳模型
        if self.best_model_state is not None:
            self.classifier.load_state_dict(self.best_model_state)

        # 最终测试
        print("\nFinal Test Evaluation:")
        test_metrics = self.evaluate(test_loader)
        print(f"Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")

        # 保存测试结果
        history["test_loss"] = test_metrics["loss"]
        history["test_accuracy"] = test_metrics["accuracy"]
        history["test_probabilities"] = test_metrics["probabilities"]
        history["test_predictions"] = test_metrics["predictions"]
        history["test_labels"] = test_metrics["labels"]

        return history

    def save_checkpoint(self, save_dir: str, tag: str):
        """保存检查点"""
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
        """加载检查点"""
        checkpoint = torch.load(path, map_location=self.device)

        self.classifier.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]

        if checkpoint.get("feature_mean") is not None:
            self.feature_mean = checkpoint["feature_mean"]
            self.feature_std = checkpoint["feature_std"]


class FeatureCollector:
    """
    特征收集器

    从目标模型和影子模型系统收集攻击特征
    """

    def __init__(
            self,
            target_model: nn.Module,
            shadow_system: nn.Module,
            feature_extractor: nn.Module,
            device: str = "cuda",
    ):
        """
        Args:
            target_model: 目标模型
            shadow_system: 影子模型系统 (多专家)
            feature_extractor: 攻击特征提取器
            device: 计算设备
        """
        self.target_model = target_model
        self.shadow_system = shadow_system
        self.feature_extractor = feature_extractor
        self.device = torch.device(device)

        # 移动到设备
        self.target_model.to(self.device).eval()
        self.shadow_system.to(self.device).eval()

    @torch.no_grad()
    def collect_features(
            self,
            dataloader: DataLoader,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        收集攻击特征

        Args:
            dataloader: 数据加载器

        Returns:
            features: 特征矩阵 [N, feature_dim]
            labels: 标签 [N]
        """
        all_features = []
        all_labels = []

        for batch in tqdm(dataloader, desc="Collecting features"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            labels = batch["label"]

            # 获取目标模型输出
            target_outputs = self.target_model(input_ids, attention_mask)
            if isinstance(target_outputs, dict):
                target_logits = target_outputs["logits"]
            else:
                target_logits = target_outputs
            target_probs = F.softmax(target_logits, dim=-1)

            # 计算目标模型损失
            shift_logits = target_logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            target_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction='none'
            ).view(input_ids.size(0), -1).mean(dim=-1)

            # 获取影子系统输出
            shadow_outputs = self.shadow_system.forward_all_experts(
                input_ids, attention_mask
            )
            expert_probs = [
                F.softmax(out["logits"] if isinstance(out, dict) else out, dim=-1)
                for out in shadow_outputs
            ]

            # 获取路由权重 (如果影子系统有路由)
            if hasattr(self.shadow_system, 'router'):
                # 假设有特征提取器
                routing_weights = torch.ones(input_ids.size(0), len(expert_probs)).to(self.device)
                routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
            else:
                routing_weights = torch.ones(input_ids.size(0), len(expert_probs)).to(self.device)
                routing_weights = routing_weights / len(expert_probs)

            # 提取攻击特征
            features = self.feature_extractor(
                target_probs, target_loss, input_ids,
                expert_probs, routing_weights
            )

            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

        return np.vstack(all_features), np.concatenate(all_labels)


class AttackPipeline:
    """
    完整的攻击流水线

    整合特征收集、模型训练和攻击执行
    """

    def __init__(
            self,
            target_model: nn.Module,
            shadow_system: nn.Module,
            attack_classifier: nn.Module,
            feature_extractor: nn.Module,
            config: AttackTrainingConfig,
    ):
        """
        Args:
            target_model: 目标模型
            shadow_system: 影子模型系统
            attack_classifier: 攻击分类器
            feature_extractor: 特征提取器
            config: 训练配置
        """
        self.target_model = target_model
        self.shadow_system = shadow_system
        self.attack_classifier = attack_classifier
        self.feature_extractor = feature_extractor
        self.config = config

        self.device = torch.device(config.device)

        # 特征收集器
        self.collector = FeatureCollector(
            target_model, shadow_system,
            feature_extractor, config.device
        )

        # 攻击训练器
        self.trainer = AttackTrainer(attack_classifier, config)

    def run(
            self,
            train_dataloader: DataLoader,
            attack_dataloader: DataLoader,
            save_dir: Optional[str] = None,
    ) -> Dict:
        """
        运行完整攻击流水线

        Args:
            train_dataloader: 训练数据加载器 (用于训练攻击分类器)
            attack_dataloader: 攻击数据加载器 (用于最终评估)
            save_dir: 保存目录

        Returns:
            results: 攻击结果
        """
        results = {}

        # 1. 收集训练特征
        print("\n=== Phase 1: Collecting Training Features ===")
        train_features, train_labels = self.collector.collect_features(train_dataloader)
        print(f"Collected {len(train_features)} training samples")
        results["train_features_shape"] = train_features.shape

        # 2. 训练攻击分类器
        print("\n=== Phase 2: Training Attack Classifier ===")
        attack_save_dir = os.path.join(save_dir, "attack") if save_dir else None
        history = self.trainer.train(train_features, train_labels, attack_save_dir)
        results["training_history"] = history

        # 3. 收集攻击特征
        print("\n=== Phase 3: Collecting Attack Features ===")
        attack_features, attack_labels = self.collector.collect_features(attack_dataloader)
        print(f"Collected {len(attack_features)} attack samples")

        # 4. 执行攻击
        print("\n=== Phase 4: Executing Attack ===")

        # 标准化
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
    # 测试代码
    print("Testing Attack Training Module...")

    from src.models import AttackClassifier

    # 创建测试数据
    np.random.seed(42)
    num_samples = 1000
    feature_dim = 45

    features = np.random.randn(num_samples, feature_dim).astype(np.float32)
    # 创建有一定可分性的标签
    labels = (features[:, 0] + features[:, 1] + np.random.randn(num_samples) * 0.5 > 0).astype(np.float32)

    print(f"Features shape: {features.shape}")
    print(f"Labels distribution: {labels.mean():.2f}")

    # 创建攻击分类器
    classifier = AttackClassifier(input_dim=feature_dim)

    # 创建配置
    config = AttackTrainingConfig(
        num_epochs=10,
        batch_size=64,
        learning_rate=1e-3,
        device="cpu",
    )

    # 创建训练器
    trainer = AttackTrainer(classifier, config)

    # 训练
    history = trainer.train(features, labels)

    print(f"\nFinal Test Accuracy: {history['test_accuracy']:.4f}")
    print(f"Best Validation Loss: {trainer.best_val_loss:.4f}")

    print("\nAll tests passed!")