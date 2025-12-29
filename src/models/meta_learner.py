"""
Meta Learner Module
===================

元学习决策器，实现论文阶段III中的最终决策融合机制。
将多专家输出和路由信息融合，生成最终的成员推理预测。

论文参考:
    - 阶段III: 智能集成机制构建与训练 (Section 4)
    - 最终概率: p_final(x) = MetaLearner(f_meta(x))
"""

from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MetaFeatureBuilder(nn.Module):
    """
    元特征构建器
    
    整合来自多个来源的特征:
        1. 专家输出统计特征
        2. 路由权重特征
        3. 输入文本特征
        4. 专家间差异特征
    """
    
    def __init__(
        self,
        num_experts: int = 8,
        expert_output_dim: int = 50000,  # vocab size
        input_feature_dim: int = 768,
        routing_dim: int = 8,
    ):
        """
        Args:
            num_experts: 专家数量
            expert_output_dim: 专家输出维度 (词汇表大小)
            input_feature_dim: 输入特征维度
            routing_dim: 路由特征维度
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.expert_output_dim = expert_output_dim
        self.input_feature_dim = input_feature_dim
        
        # 专家输出特征提取 (每个专家提取固定维度的统计特征)
        self.expert_feature_dim = 16  # 每个专家的特征维度
        
        # 专家输出压缩网络
        self.expert_compressors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(expert_output_dim, 256),
                nn.ReLU(),
                nn.Linear(256, self.expert_feature_dim),
            )
            for _ in range(num_experts)
        ])
        
        # 输入特征压缩
        self.input_compressor = nn.Sequential(
            nn.Linear(input_feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )
        
        # 计算总特征维度
        self.total_feature_dim = (
            num_experts * self.expert_feature_dim +  # 压缩后的专家输出
            num_experts +  # 路由权重
            64 +  # 压缩后的输入特征
            num_experts * (num_experts - 1) // 2 +  # 专家对差异
            num_experts * 4  # 专家统计特征 (mean, std, max, entropy)
        )
        
    def compute_expert_statistics(
        self,
        expert_probs: List[Tensor],
    ) -> Tensor:
        """
        计算专家输出的统计特征
        
        Args:
            expert_probs: 专家概率分布列表
            
        Returns:
            统计特征 [batch, num_experts * 4]
        """
        stats_list = []
        
        for probs in expert_probs:
            # probs: [batch, seq, vocab] -> 取最后一个位置
            if probs.dim() == 3:
                probs = probs[:, -1, :]  # [batch, vocab]
            
            # 计算统计量
            mean_prob = probs.mean(dim=-1, keepdim=True)
            std_prob = probs.std(dim=-1, keepdim=True)
            max_prob = probs.max(dim=-1, keepdim=True)[0]
            entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1, keepdim=True)
            
            stats = torch.cat([mean_prob, std_prob, max_prob, entropy], dim=-1)
            stats_list.append(stats)
            
        return torch.cat(stats_list, dim=-1)
    
    def compute_pairwise_differences(
        self,
        expert_probs: List[Tensor],
    ) -> Tensor:
        """
        计算专家对之间的差异
        
        Args:
            expert_probs: 专家概率分布列表
            
        Returns:
            差异特征 [batch, num_pairs]
        """
        differences = []
        
        for i in range(self.num_experts):
            for j in range(i + 1, self.num_experts):
                # 取最后一个位置的概率
                probs_i = expert_probs[i]
                probs_j = expert_probs[j]
                
                if probs_i.dim() == 3:
                    probs_i = probs_i[:, -1, :]
                    probs_j = probs_j[:, -1, :]
                
                # 计算KL散度作为差异度量
                kl_div = F.kl_div(
                    probs_i.log(),
                    probs_j,
                    reduction='none'
                ).sum(dim=-1, keepdim=True)
                
                differences.append(kl_div)
                
        return torch.cat(differences, dim=-1)
    
    def forward(
        self,
        expert_outputs: List[Tensor],
        routing_weights: Tensor,
        input_features: Tensor,
    ) -> Tensor:
        """
        前向传播 - 构建元特征
        
        Args:
            expert_outputs: 专家输出概率列表 [batch, seq, vocab]
            routing_weights: 路由权重 [batch, num_experts]
            input_features: 输入特征 [batch, input_feature_dim]
            
        Returns:
            meta_features: 元特征向量 [batch, total_feature_dim]
        """
        features_list = []
        
        # 1. 压缩专家输出
        for i, (output, compressor) in enumerate(zip(expert_outputs, self.expert_compressors)):
            if output.dim() == 3:
                output = output[:, -1, :]  # [batch, vocab]
            compressed = compressor(output)
            features_list.append(compressed)
            
        # 2. 路由权重作为特征
        features_list.append(routing_weights)
        
        # 3. 压缩输入特征
        compressed_input = self.input_compressor(input_features)
        features_list.append(compressed_input)
        
        # 4. 专家对差异
        pairwise_diff = self.compute_pairwise_differences(expert_outputs)
        features_list.append(pairwise_diff)
        
        # 5. 专家统计特征
        expert_stats = self.compute_expert_statistics(expert_outputs)
        features_list.append(expert_stats)
        
        # 合并所有特征
        meta_features = torch.cat(features_list, dim=-1)
        
        return meta_features


class MetaLearner(nn.Module):
    """
    元学习决策器
    
    基于元特征做出最终的成员推理预测决策。
    
    论文公式:
        p_final(x) = MetaLearner(f_meta(x))
        
    损失函数:
        L_meta = -E[y log p_final + (1-y) log(1-p_final)] + λ_reg ||θ_meta||²
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128, 64],
        activation: str = "relu",
        dropout: float = 0.2,
        output_dim: int = 1,
    ):
        """
        Args:
            input_dim: 输入特征维度
            hidden_dims: 隐藏层维度列表
            activation: 激活函数
            dropout: Dropout率
            output_dim: 输出维度 (1 for binary classification)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 激活函数
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        # 构建MLP层
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                self.activation,
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
            
        # 输出层 (不使用激活，用于logits)
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, meta_features: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            meta_features: 元特征 [batch, input_dim]
            
        Returns:
            logits: 输出logits [batch, output_dim]
        """
        return self.network(meta_features)
    
    def predict_proba(self, meta_features: Tensor) -> Tensor:
        """
        预测概率
        
        Args:
            meta_features: 元特征
            
        Returns:
            probabilities: 成员概率 [batch, 1]
        """
        logits = self.forward(meta_features)
        return torch.sigmoid(logits)
    
    def predict(self, meta_features: Tensor, threshold: float = 0.5) -> Tensor:
        """
        预测标签
        
        Args:
            meta_features: 元特征
            threshold: 决策阈值
            
        Returns:
            predictions: 预测标签 [batch]
        """
        proba = self.predict_proba(meta_features)
        return (proba >= threshold).long().squeeze(-1)


class EnsembleDecisionMaker(nn.Module):
    """
    集成决策模块
    
    整合MetaFeatureBuilder和MetaLearner，
    并管理整个集成决策流程。
    """
    
    def __init__(
        self,
        num_experts: int = 8,
        expert_output_dim: int = 50000,
        input_feature_dim: int = 768,
        meta_hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.2,
    ):
        """
        Args:
            num_experts: 专家数量
            expert_output_dim: 专家输出维度
            input_feature_dim: 输入特征维度
            meta_hidden_dims: 元学习器隐藏层维度
            dropout: Dropout率
        """
        super().__init__()
        
        self.num_experts = num_experts
        
        # 元特征构建器
        self.feature_builder = MetaFeatureBuilder(
            num_experts=num_experts,
            expert_output_dim=expert_output_dim,
            input_feature_dim=input_feature_dim,
        )
        
        # 元学习器
        self.meta_learner = MetaLearner(
            input_dim=self.feature_builder.total_feature_dim,
            hidden_dims=meta_hidden_dims,
            dropout=dropout,
        )
        
    def forward(
        self,
        expert_outputs: List[Tensor],
        routing_weights: Tensor,
        input_features: Tensor,
    ) -> Tensor:
        """
        前向传播
        
        Args:
            expert_outputs: 专家输出列表
            routing_weights: 路由权重
            input_features: 输入特征
            
        Returns:
            logits: 成员推理logits
        """
        # 构建元特征
        meta_features = self.feature_builder(
            expert_outputs, routing_weights, input_features
        )
        
        # 元学习器预测
        logits = self.meta_learner(meta_features)
        
        return logits
    
    def predict_membership(
        self,
        expert_outputs: List[Tensor],
        routing_weights: Tensor,
        input_features: Tensor,
        threshold: float = 0.5,
    ) -> Tuple[Tensor, Tensor]:
        """
        预测成员关系
        
        Args:
            expert_outputs: 专家输出
            routing_weights: 路由权重
            input_features: 输入特征
            threshold: 决策阈值
            
        Returns:
            predictions: 预测标签
            probabilities: 成员概率
        """
        logits = self.forward(expert_outputs, routing_weights, input_features)
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).long().squeeze(-1)
        
        return predictions, probabilities


class WeightedEnsemble(nn.Module):
    """
    加权集成模块
    
    使用学习的权重来组合多个专家的预测
    """
    
    def __init__(
        self,
        num_experts: int = 8,
        combine_method: str = "weighted_sum",
    ):
        """
        Args:
            num_experts: 专家数量
            combine_method: 组合方法 ("weighted_sum", "attention", "gating")
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.combine_method = combine_method
        
        if combine_method == "weighted_sum":
            # 可学习的静态权重
            self.weights = nn.Parameter(torch.ones(num_experts) / num_experts)
        elif combine_method == "gating":
            # 门控网络动态权重
            self.gate = nn.Sequential(
                nn.Linear(num_experts, num_experts * 2),
                nn.ReLU(),
                nn.Linear(num_experts * 2, num_experts),
                nn.Softmax(dim=-1),
            )
            
    def forward(
        self,
        expert_predictions: Tensor,
        routing_weights: Optional[Tensor] = None,
    ) -> Tensor:
        """
        前向传播
        
        Args:
            expert_predictions: 专家预测 [batch, num_experts]
            routing_weights: 路由权重 (可选)
            
        Returns:
            combined: 组合后的预测 [batch, 1]
        """
        if self.combine_method == "weighted_sum":
            weights = F.softmax(self.weights, dim=0)
            combined = (expert_predictions * weights).sum(dim=-1, keepdim=True)
        elif self.combine_method == "gating":
            weights = self.gate(expert_predictions)
            combined = (expert_predictions * weights).sum(dim=-1, keepdim=True)
        elif self.combine_method == "routing":
            if routing_weights is None:
                raise ValueError("routing_weights required for routing combine method")
            combined = (expert_predictions * routing_weights).sum(dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown combine method: {self.combine_method}")
            
        return combined


class AttentionFusion(nn.Module):
    """
    基于注意力的特征融合
    
    使用自注意力机制融合多专家特征
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_experts: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        """
        Args:
            feature_dim: 特征维度
            num_experts: 专家数量
            num_heads: 注意力头数
            dropout: Dropout率
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.feature_dim = feature_dim
        
        # 多头自注意力
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # 输出投影
        self.output_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
        )
        
    def forward(self, expert_features: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            expert_features: 专家特征 [batch, num_experts, feature_dim]
            
        Returns:
            fused: 融合后的特征 [batch, feature_dim]
        """
        # 自注意力
        attn_output, _ = self.attention(
            expert_features, expert_features, expert_features
        )
        
        # 残差连接
        attn_output = attn_output + expert_features
        
        # 全局平均池化
        pooled = attn_output.mean(dim=1)
        
        # 输出投影
        output = self.output_proj(pooled)
        
        return output


if __name__ == "__main__":
    # 测试代码
    print("Testing Meta Learner Module...")
    
    batch_size = 4
    num_experts = 8
    vocab_size = 50000
    input_feature_dim = 768
    
    # 创建测试数据
    expert_outputs = [
        torch.randn(batch_size, vocab_size).softmax(dim=-1)
        for _ in range(num_experts)
    ]
    routing_weights = torch.randn(batch_size, num_experts).softmax(dim=-1)
    input_features = torch.randn(batch_size, input_feature_dim)
    
    # 测试MetaFeatureBuilder
    print("Testing MetaFeatureBuilder...")
    feature_builder = MetaFeatureBuilder(
        num_experts=num_experts,
        expert_output_dim=vocab_size,
        input_feature_dim=input_feature_dim,
    )
    meta_features = feature_builder(expert_outputs, routing_weights, input_features)
    print(f"Meta features shape: {meta_features.shape}")
    print(f"Total feature dim: {feature_builder.total_feature_dim}")
    
    # 测试MetaLearner
    print("\nTesting MetaLearner...")
    meta_learner = MetaLearner(
        input_dim=feature_builder.total_feature_dim,
        hidden_dims=[256, 128, 64],
    )
    logits = meta_learner(meta_features)
    proba = meta_learner.predict_proba(meta_features)
    predictions = meta_learner.predict(meta_features)
    
    print(f"Logits shape: {logits.shape}")
    print(f"Probabilities shape: {proba.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    # 测试EnsembleDecisionMaker
    print("\nTesting EnsembleDecisionMaker...")
    ensemble = EnsembleDecisionMaker(
        num_experts=num_experts,
        expert_output_dim=vocab_size,
        input_feature_dim=input_feature_dim,
    )
    preds, probs = ensemble.predict_membership(
        expert_outputs, routing_weights, input_features
    )
    print(f"Ensemble predictions shape: {preds.shape}")
    print(f"Ensemble probabilities shape: {probs.shape}")
    
    # 测试WeightedEnsemble
    print("\nTesting WeightedEnsemble...")
    weighted_ensemble = WeightedEnsemble(num_experts)
    expert_preds = torch.randn(batch_size, num_experts).sigmoid()
    combined = weighted_ensemble(expert_preds)
    print(f"Combined prediction shape: {combined.shape}")
    
    # 测试AttentionFusion
    print("\nTesting AttentionFusion...")
    fusion = AttentionFusion(feature_dim=256, num_experts=num_experts)
    expert_features = torch.randn(batch_size, num_experts, 256)
    fused = fusion(expert_features)
    print(f"Fused features shape: {fused.shape}")
    
    print("\nAll tests passed!")
