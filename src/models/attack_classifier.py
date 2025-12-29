"""
Attack Classifier Module
========================

成员推理攻击分类器，实现论文阶段IV中的攻击执行模块。
基于提取的特征向量进行二分类，判断样本是否为训练集成员。

论文参考:
    - 阶段IV: 黑盒成员推理攻击执行 (Section 5)
    - 特征维度: 目标模型12维 + 多专家系统28维 + 交互特征5维 = 45维
    - 分类器架构: 三层MLP (128-64-1)
"""

from typing import Optional, Tuple, Dict, List, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AttackFeatureExtractor(nn.Module):
    """
    攻击特征提取器
    
    从目标模型和多专家影子模型系统中提取用于成员推理的特征。
    
    论文特征维度:
        - 目标模型特征: 12维
        - 多专家系统特征: 28维
        - 交互特征: 5维
        - 总计: 45维
    """
    
    def __init__(
        self,
        target_feature_dim: int = 12,
        multi_expert_feature_dim: int = 28,
        interaction_feature_dim: int = 5,
        num_experts: int = 8,
    ):
        """
        Args:
            target_feature_dim: 目标模型特征维度
            multi_expert_feature_dim: 多专家系统特征维度
            interaction_feature_dim: 交互特征维度
            num_experts: 专家数量
        """
        super().__init__()
        
        self.target_feature_dim = target_feature_dim
        self.multi_expert_feature_dim = multi_expert_feature_dim
        self.interaction_feature_dim = interaction_feature_dim
        self.num_experts = num_experts
        
        self.total_feature_dim = (
            target_feature_dim + 
            multi_expert_feature_dim + 
            interaction_feature_dim
        )
        
    def extract_target_features(
        self,
        target_probs: Tensor,
        target_loss: Tensor,
        input_ids: Tensor,
    ) -> Tensor:
        """
        提取目标模型特征 (12维)
        
        特征包括:
            1. Loss值 (1维)
            2. Perplexity (1维)
            3. Top-1概率统计 (3维: mean, std, max)
            4. Entropy统计 (3维: mean, std, min)
            5. Token-level特征 (4维: 首位概率等)
            
        Args:
            target_probs: 目标模型输出概率 [batch, seq, vocab]
            target_loss: 目标模型损失 [batch]
            input_ids: 输入token ID [batch, seq]
            
        Returns:
            target_features: 目标模型特征 [batch, 12]
        """
        batch_size = target_probs.shape[0]
        features = []
        
        # 1. Loss值
        features.append(target_loss.unsqueeze(-1))  # [batch, 1]
        
        # 2. Perplexity
        perplexity = torch.exp(target_loss).unsqueeze(-1)
        features.append(perplexity)  # [batch, 1]
        
        # 3. Top-1概率统计
        top1_probs = target_probs.max(dim=-1)[0]  # [batch, seq]
        features.append(top1_probs.mean(dim=-1, keepdim=True))
        features.append(top1_probs.std(dim=-1, keepdim=True))
        features.append(top1_probs.max(dim=-1, keepdim=True)[0])
        
        # 4. Entropy统计
        entropy = -(target_probs * torch.log(target_probs + 1e-10)).sum(dim=-1)  # [batch, seq]
        features.append(entropy.mean(dim=-1, keepdim=True))
        features.append(entropy.std(dim=-1, keepdim=True))
        features.append(entropy.min(dim=-1, keepdim=True)[0])
        
        # 5. Token-level特征
        # 获取每个位置的正确token概率
        seq_length = target_probs.shape[1]
        if seq_length > 1:
            # Shift for causal LM
            shifted_probs = target_probs[:, :-1, :]  # [batch, seq-1, vocab]
            shifted_ids = input_ids[:, 1:]  # [batch, seq-1]
            
            # 获取正确token的概率
            correct_probs = shifted_probs.gather(
                dim=-1, 
                index=shifted_ids.unsqueeze(-1)
            ).squeeze(-1)  # [batch, seq-1]
            
            features.append(correct_probs[:, 0:1])  # 首位概率
            features.append(correct_probs[:, -1:])  # 末位概率
            features.append(correct_probs.mean(dim=-1, keepdim=True))  # 平均概率
            features.append(correct_probs.min(dim=-1, keepdim=True)[0])  # 最小概率
        else:
            # 如果序列长度为1，使用padding
            features.extend([torch.zeros(batch_size, 1, device=target_probs.device)] * 4)
        
        return torch.cat(features, dim=-1)
    
    def extract_multi_expert_features(
        self,
        expert_probs_list: List[Tensor],
        routing_weights: Tensor,
    ) -> Tensor:
        """
        提取多专家系统特征 (28维)
        
        特征包括:
            1. 每个专家的Top-1概率均值 (8维)
            2. 每个专家的Entropy均值 (8维)
            3. 路由权重 (8维)
            4. 专家间一致性 (4维)
            
        Args:
            expert_probs_list: 专家输出概率列表 [batch, seq, vocab] x num_experts
            routing_weights: 路由权重 [batch, num_experts]
            
        Returns:
            expert_features: 多专家系统特征 [batch, 28]
        """
        features = []
        
        expert_top1_means = []
        expert_entropy_means = []
        
        for expert_probs in expert_probs_list:
            # Top-1概率均值
            top1_probs = expert_probs.max(dim=-1)[0]  # [batch, seq]
            top1_mean = top1_probs.mean(dim=-1, keepdim=True)
            expert_top1_means.append(top1_mean)
            
            # Entropy均值
            entropy = -(expert_probs * torch.log(expert_probs + 1e-10)).sum(dim=-1)
            entropy_mean = entropy.mean(dim=-1, keepdim=True)
            expert_entropy_means.append(entropy_mean)
        
        # 1. 专家Top-1概率均值 (8维)
        features.extend(expert_top1_means)
        
        # 2. 专家Entropy均值 (8维)
        features.extend(expert_entropy_means)
        
        # 3. 路由权重 (8维)
        features.append(routing_weights)
        
        # 4. 专家间一致性特征 (4维)
        # 堆叠专家Top-1概率
        stacked_top1 = torch.cat(expert_top1_means, dim=-1)  # [batch, num_experts]
        
        # 专家间方差
        expert_variance = stacked_top1.var(dim=-1, keepdim=True)
        features.append(expert_variance)
        
        # Top-k专家一致性 (前3个专家权重之和)
        top3_weights = routing_weights.topk(3, dim=-1)[0].sum(dim=-1, keepdim=True)
        features.append(top3_weights)
        
        # 专家极差
        expert_range = stacked_top1.max(dim=-1, keepdim=True)[0] - stacked_top1.min(dim=-1, keepdim=True)[0]
        features.append(expert_range)
        
        # 熵加权一致性
        stacked_entropy = torch.cat(expert_entropy_means, dim=-1)
        weighted_entropy = (stacked_entropy * routing_weights).sum(dim=-1, keepdim=True)
        features.append(weighted_entropy)
        
        return torch.cat(features, dim=-1)
    
    def extract_interaction_features(
        self,
        target_probs: Tensor,
        expert_probs_list: List[Tensor],
        routing_weights: Tensor,
    ) -> Tensor:
        """
        提取交互特征 (5维)
        
        特征包括:
            1. 目标模型与加权专家预测的KL散度
            2. 目标模型与最佳专家的余弦相似度
            3. 预测差异的统计量
            
        Args:
            target_probs: 目标模型概率 [batch, seq, vocab]
            expert_probs_list: 专家概率列表
            routing_weights: 路由权重
            
        Returns:
            interaction_features: 交互特征 [batch, 5]
        """
        features = []
        
        # 计算加权专家预测
        weighted_expert_probs = torch.zeros_like(target_probs)
        for i, expert_probs in enumerate(expert_probs_list):
            weighted_expert_probs += routing_weights[:, i:i+1, None] * expert_probs
            
        # 1. KL散度 (目标 || 加权专家)
        kl_div = F.kl_div(
            weighted_expert_probs.log(),
            target_probs,
            reduction='none'
        ).sum(dim=-1).mean(dim=-1, keepdim=True)
        features.append(kl_div)
        
        # 2. 反向KL散度 (加权专家 || 目标)
        reverse_kl = F.kl_div(
            target_probs.log(),
            weighted_expert_probs,
            reduction='none'
        ).sum(dim=-1).mean(dim=-1, keepdim=True)
        features.append(reverse_kl)
        
        # 3. 与最佳专家的相似度
        best_expert_idx = routing_weights.argmax(dim=-1)  # [batch]
        batch_size = target_probs.shape[0]
        
        # 简化计算：使用最后一个位置
        target_last = target_probs[:, -1, :]  # [batch, vocab]
        best_expert_probs = torch.stack([
            expert_probs_list[best_expert_idx[i].item()][i, -1, :]
            for i in range(batch_size)
        ])  # [batch, vocab]
        
        # 余弦相似度
        cosine_sim = F.cosine_similarity(target_last, best_expert_probs, dim=-1)
        features.append(cosine_sim.unsqueeze(-1))
        
        # 4. 预测差异均值
        diff = (target_probs - weighted_expert_probs).abs().mean(dim=(-1, -2), keepdim=True)
        features.append(diff.squeeze(-1))
        
        # 5. 预测差异标准差
        diff_std = (target_probs - weighted_expert_probs).abs().std(dim=(-1, -2), keepdim=True)
        features.append(diff_std.squeeze(-1))
        
        return torch.cat(features, dim=-1)
    
    def forward(
        self,
        target_probs: Tensor,
        target_loss: Tensor,
        input_ids: Tensor,
        expert_probs_list: List[Tensor],
        routing_weights: Tensor,
    ) -> Tensor:
        """
        提取完整攻击特征向量
        
        Args:
            target_probs: 目标模型概率
            target_loss: 目标模型损失
            input_ids: 输入token IDs
            expert_probs_list: 专家概率列表
            routing_weights: 路由权重
            
        Returns:
            attack_features: 攻击特征 [batch, 45]
        """
        # 提取三类特征
        target_features = self.extract_target_features(
            target_probs, target_loss, input_ids
        )
        
        multi_expert_features = self.extract_multi_expert_features(
            expert_probs_list, routing_weights
        )
        
        interaction_features = self.extract_interaction_features(
            target_probs, expert_probs_list, routing_weights
        )
        
        # 合并所有特征
        attack_features = torch.cat([
            target_features,
            multi_expert_features,
            interaction_features,
        ], dim=-1)
        
        return attack_features


class AttackClassifier(nn.Module):
    """
    攻击分类器 - 三层MLP
    
    论文架构 (公式6-8):
        h1 = ReLU(W1 * f_attack(x) + b1), W1 ∈ R^{128×45}
        h2 = ReLU(W2 * h1 + b2), W2 ∈ R^{64×128}
        p_attack(x) = σ(W3 * h2 + b3), W3 ∈ R^{1×64}
    """
    
    def __init__(
        self,
        input_dim: int = 45,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.3,
    ):
        """
        Args:
            input_dim: 输入特征维度 (论文中为45)
            hidden_dims: 隐藏层维度 [128, 64]
            dropout: Dropout率
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # 第一层: 45 -> 128
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 第二层: 128 -> 64
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # 输出层: 64 -> 1
        self.output_layer = nn.Linear(hidden_dims[1], 1)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: 攻击特征 [batch, input_dim]
            
        Returns:
            logits: 输出logits [batch, 1]
        """
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        logits = self.output_layer(h2)
        return logits
    
    def predict_proba(self, x: Tensor) -> Tensor:
        """预测成员概率"""
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def predict(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        """预测成员标签"""
        proba = self.predict_proba(x)
        return (proba >= threshold).long().squeeze(-1)


class MembershipInferenceAttack(nn.Module):
    """
    完整的成员推理攻击模块
    
    整合特征提取和分类器，提供端到端的攻击接口。
    """
    
    def __init__(
        self,
        num_experts: int = 8,
        target_feature_dim: int = 12,
        multi_expert_feature_dim: int = 28,
        interaction_feature_dim: int = 5,
        classifier_hidden_dims: List[int] = [128, 64],
        dropout: float = 0.3,
    ):
        """
        Args:
            num_experts: 专家数量
            target_feature_dim: 目标模型特征维度
            multi_expert_feature_dim: 多专家特征维度
            interaction_feature_dim: 交互特征维度
            classifier_hidden_dims: 分类器隐藏层维度
            dropout: Dropout率
        """
        super().__init__()
        
        # 特征提取器
        self.feature_extractor = AttackFeatureExtractor(
            target_feature_dim=target_feature_dim,
            multi_expert_feature_dim=multi_expert_feature_dim,
            interaction_feature_dim=interaction_feature_dim,
            num_experts=num_experts,
        )
        
        # 攻击分类器
        self.classifier = AttackClassifier(
            input_dim=self.feature_extractor.total_feature_dim,
            hidden_dims=classifier_hidden_dims,
            dropout=dropout,
        )
        
        # 特征标准化参数
        self.register_buffer('feature_mean', None)
        self.register_buffer('feature_std', None)
        
    def fit_scaler(self, features: Tensor):
        """
        拟合特征标准化器
        
        论文5.2节: f'_attack(x) = (f_attack(x) - μ) / σ
        
        Args:
            features: 训练特征 [N, feature_dim]
        """
        self.feature_mean = features.mean(dim=0)
        self.feature_std = features.std(dim=0).clamp(min=1e-6)
        
    def normalize_features(self, features: Tensor) -> Tensor:
        """Z-score标准化"""
        if self.feature_mean is None:
            return features
        return (features - self.feature_mean) / self.feature_std
    
    def extract_features(
        self,
        target_probs: Tensor,
        target_loss: Tensor,
        input_ids: Tensor,
        expert_probs_list: List[Tensor],
        routing_weights: Tensor,
    ) -> Tensor:
        """提取并标准化特征"""
        features = self.feature_extractor(
            target_probs, target_loss, input_ids,
            expert_probs_list, routing_weights
        )
        return self.normalize_features(features)
    
    def forward(
        self,
        target_probs: Tensor,
        target_loss: Tensor,
        input_ids: Tensor,
        expert_probs_list: List[Tensor],
        routing_weights: Tensor,
    ) -> Tensor:
        """
        端到端前向传播
        
        Args:
            target_probs: 目标模型概率
            target_loss: 目标模型损失
            input_ids: 输入token IDs
            expert_probs_list: 专家概率列表
            routing_weights: 路由权重
            
        Returns:
            logits: 攻击logits
        """
        features = self.extract_features(
            target_probs, target_loss, input_ids,
            expert_probs_list, routing_weights
        )
        return self.classifier(features)
    
    def attack(
        self,
        target_probs: Tensor,
        target_loss: Tensor,
        input_ids: Tensor,
        expert_probs_list: List[Tensor],
        routing_weights: Tensor,
        threshold: float = 0.5,
    ) -> Tuple[Tensor, Tensor]:
        """
        执行攻击
        
        Args:
            ... (同forward)
            threshold: 决策阈值
            
        Returns:
            predictions: 预测标签 [batch]
            probabilities: 成员概率 [batch, 1]
        """
        logits = self.forward(
            target_probs, target_loss, input_ids,
            expert_probs_list, routing_weights
        )
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).long().squeeze(-1)
        
        return predictions, probabilities


class BaselineAttacks:
    """
    基线攻击方法集合
    
    实现用于对比的传统攻击方法
    """
    
    @staticmethod
    def loss_based_attack(
        loss: Tensor,
        threshold: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        基于损失的攻击
        
        思想: 训练样本的损失通常更低
        
        Args:
            loss: 模型损失 [batch]
            threshold: 决策阈值
            
        Returns:
            predictions: 预测标签
            scores: 攻击分数 (负损失)
        """
        scores = -loss  # 损失越低，越可能是成员
        
        if threshold is None:
            threshold = scores.median()
            
        predictions = (scores >= threshold).long()
        return predictions, scores
    
    @staticmethod
    def likelihood_ratio_attack(
        target_loss: Tensor,
        reference_loss: Tensor,
        threshold: float = 0.0,
    ) -> Tuple[Tensor, Tensor]:
        """
        似然比攻击
        
        思想: 比较目标模型和参考模型的损失差异
        
        Args:
            target_loss: 目标模型损失
            reference_loss: 参考模型损失
            threshold: 决策阈值
            
        Returns:
            predictions: 预测标签
            scores: 似然比分数
        """
        scores = reference_loss - target_loss  # 差异越大，越可能是成员
        predictions = (scores >= threshold).long()
        return predictions, scores
    
    @staticmethod
    def min_k_attack(
        probs: Tensor,
        k_ratio: float = 0.2,
        threshold: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Min-K%攻击
        
        思想: 检查最低概率token的概率分布
        
        Args:
            probs: 正确token的概率 [batch, seq]
            k_ratio: 选取最低k%的比例
            threshold: 决策阈值
            
        Returns:
            predictions: 预测标签
            scores: Min-K分数
        """
        k = max(1, int(probs.shape[1] * k_ratio))
        min_k_probs = probs.topk(k, dim=1, largest=False)[0]  # [batch, k]
        scores = min_k_probs.mean(dim=1)  # 平均最低k%概率
        
        if threshold is None:
            threshold = scores.median()
            
        # 训练样本的min-k概率应该更高
        predictions = (scores >= threshold).long()
        return predictions, scores
    
    @staticmethod
    def zlib_attack(
        loss: Tensor,
        zlib_entropy: Tensor,
        threshold: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Zlib攻击
        
        思想: 结合模型困惑度和文本压缩率
        
        Args:
            loss: 模型损失
            zlib_entropy: zlib压缩后的熵
            threshold: 决策阈值
            
        Returns:
            predictions: 预测标签
            scores: 归一化困惑度分数
        """
        # 使用zlib熵归一化困惑度
        scores = -loss / (zlib_entropy + 1e-10)
        
        if threshold is None:
            threshold = scores.median()
            
        predictions = (scores >= threshold).long()
        return predictions, scores
    
    @staticmethod
    def neighborhood_attack(
        original_loss: Tensor,
        perturbed_losses: Tensor,
        threshold: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        邻域攻击
        
        思想: 训练样本对扰动更敏感
        
        Args:
            original_loss: 原始样本损失 [batch]
            perturbed_losses: 扰动样本损失 [batch, num_perturbations]
            threshold: 决策阈值
            
        Returns:
            predictions: 预测标签
            scores: 损失差异分数
        """
        # 计算扰动后的平均损失增加
        mean_perturbed_loss = perturbed_losses.mean(dim=1)
        scores = mean_perturbed_loss - original_loss
        
        if threshold is None:
            threshold = scores.median()
            
        # 训练样本的损失增加更明显
        predictions = (scores >= threshold).long()
        return predictions, scores


if __name__ == "__main__":
    # 测试代码
    print("Testing Attack Classifier Module...")
    
    batch_size = 8
    seq_length = 128
    vocab_size = 50000
    num_experts = 8
    
    # 创建测试数据
    target_probs = torch.randn(batch_size, seq_length, vocab_size).softmax(dim=-1)
    target_loss = torch.rand(batch_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    expert_probs_list = [
        torch.randn(batch_size, seq_length, vocab_size).softmax(dim=-1)
        for _ in range(num_experts)
    ]
    routing_weights = torch.randn(batch_size, num_experts).softmax(dim=-1)
    
    # 测试AttackFeatureExtractor
    print("Testing AttackFeatureExtractor...")
    feature_extractor = AttackFeatureExtractor(num_experts=num_experts)
    features = feature_extractor(
        target_probs, target_loss, input_ids,
        expert_probs_list, routing_weights
    )
    print(f"Feature shape: {features.shape}")
    print(f"Expected dim: {feature_extractor.total_feature_dim}")
    
    # 测试AttackClassifier
    print("\nTesting AttackClassifier...")
    classifier = AttackClassifier(input_dim=features.shape[-1])
    logits = classifier(features)
    print(f"Logits shape: {logits.shape}")
    
    predictions = classifier.predict(features)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")
    
    # 测试MembershipInferenceAttack
    print("\nTesting MembershipInferenceAttack...")
    attack = MembershipInferenceAttack(num_experts=num_experts)
    
    # 先提取一批特征用于拟合标准化器
    attack.fit_scaler(features)
    
    preds, probs = attack.attack(
        target_probs, target_loss, input_ids,
        expert_probs_list, routing_weights
    )
    print(f"Attack predictions shape: {preds.shape}")
    print(f"Attack probabilities shape: {probs.shape}")
    
    # 测试基线攻击
    print("\nTesting Baseline Attacks...")
    
    # Loss-based
    loss_preds, loss_scores = BaselineAttacks.loss_based_attack(target_loss)
    print(f"Loss-based predictions: {loss_preds}")
    
    # Likelihood ratio
    ref_loss = torch.rand(batch_size)
    lr_preds, lr_scores = BaselineAttacks.likelihood_ratio_attack(target_loss, ref_loss)
    print(f"Likelihood ratio predictions: {lr_preds}")
    
    # Min-K
    correct_probs = torch.rand(batch_size, seq_length)
    mk_preds, mk_scores = BaselineAttacks.min_k_attack(correct_probs)
    print(f"Min-K predictions: {mk_preds}")
    
    print("\nAll tests passed!")
