"""
Router Network Module
=====================

动态路由选择网络，实现论文阶段III中的智能路由机制。
根据输入特征动态分配专家权重，实现多专家协同。

论文参考:
    - 阶段III: 智能集成机制构建与训练 (Section 4)
    - 路由权重: w_k(x) = RouteNet(φ(x))
"""

import math
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class FeatureExtractor(nn.Module):
    """
    特征提取器 - 基于BERT的文本表示提取
    
    论文公式(4): φ(x) = MeanPooling(BERT(x)) ∈ R^768
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dim: int = 768,
        pooling: str = "mean",
        freeze: bool = True,
    ):
        """
        Args:
            model_name: 预训练模型名称
            output_dim: 输出维度
            pooling: 池化方式 ("mean", "cls", "max")
            freeze: 是否冻结BERT参数
        """
        super().__init__()
        
        self.output_dim = output_dim
        self.pooling = pooling
        
        # 延迟加载BERT (避免import时加载)
        self._bert = None
        self._tokenizer = None
        self.model_name = model_name
        self.freeze = freeze
        
    def _load_bert(self):
        """延迟加载BERT模型"""
        if self._bert is None:
            from transformers import AutoModel, AutoTokenizer
            
            self._bert = AutoModel.from_pretrained(self.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            if self.freeze:
                for param in self._bert.parameters():
                    param.requires_grad = False
                    
    @property
    def bert(self):
        self._load_bert()
        return self._bert
    
    @property
    def tokenizer(self):
        self._load_bert()
        return self._tokenizer
        
    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        text: Optional[List[str]] = None,
    ) -> Tensor:
        """
        前向传播
        
        Args:
            input_ids: Token IDs [batch, seq]
            attention_mask: 注意力掩码
            text: 原始文本列表 (可选，会自动tokenize)
            
        Returns:
            特征向量 [batch, output_dim]
        """
        # 如果提供原始文本，进行tokenization
        if text is not None:
            encoded = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            )
            input_ids = encoded["input_ids"]
            attention_mask = encoded["attention_mask"]
            
            # 移动到正确的设备
            device = next(self.bert.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
        
        # BERT前向传播
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [batch, seq, hidden]
        
        # 池化操作
        if self.pooling == "mean":
            # Mean pooling with attention mask
            if attention_mask is not None:
                mask_expanded = attention_mask.unsqueeze(-1).float()
                sum_hidden = (hidden_states * mask_expanded).sum(dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                features = sum_hidden / sum_mask
            else:
                features = hidden_states.mean(dim=1)
        elif self.pooling == "cls":
            features = hidden_states[:, 0, :]
        elif self.pooling == "max":
            features = hidden_states.max(dim=1)[0]
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
            
        return features


class SimpleFeatureExtractor(nn.Module):
    """
    简化特征提取器 - 不依赖BERT，用于测试
    """
    
    def __init__(
        self,
        vocab_size: int = 50000,
        embed_dim: int = 256,
        output_dim: int = 768,
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, 
            output_dim // 2, 
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.output_dim = output_dim
        
    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """前向传播"""
        embedded = self.embedding(input_ids)
        outputs, (h_n, _) = self.lstm(embedded)
        
        # 使用最后一层的双向隐藏状态
        features = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return features


class RouterNetwork(nn.Module):
    """
    路由网络 - 动态专家选择
    
    输入文本特征，输出K个专家的权重分配。
    
    论文参考:
        - 路由权重: {w_k(x)}_{k=1}^8 = RouteNet(φ(x))
        - Top-K稀疏激活机制
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        num_experts: int = 8,
        hidden_dims: List[int] = [512, 256],
        activation: str = "gelu",
        dropout: float = 0.1,
        top_k: int = 3,
        temperature: float = 1.0,
    ):
        """
        Args:
            input_dim: 输入特征维度
            num_experts: 专家数量
            hidden_dims: 隐藏层维度列表
            activation: 激活函数
            dropout: Dropout率
            top_k: Top-K稀疏激活
            temperature: Softmax温度
        """
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        self.temperature = temperature
        
        # 激活函数
        if activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation: {activation}")
            
        # 构建MLP层
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                self.activation,
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
            
        # 输出层
        layers.append(nn.Linear(prev_dim, num_experts))
        
        self.router = nn.Sequential(*layers)
        
        # 噪声用于探索 (可选)
        self.noise_std = 0.1
        
    def forward(
        self,
        features: Tensor,
        add_noise: bool = False,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        前向传播
        
        Args:
            features: 输入特征 [batch, input_dim]
            add_noise: 是否添加噪声 (用于训练时探索)
            
        Returns:
            weights: 归一化的专家权重 [batch, num_experts]
            top_k_indices: Top-K专家索引 [batch, top_k]
            top_k_weights: Top-K专家权重 [batch, top_k]
        """
        # 计算路由logits
        logits = self.router(features)  # [batch, num_experts]
        
        # 可选：添加噪声进行探索
        if add_noise and self.training:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
            
        # 应用温度
        logits = logits / self.temperature
        
        # 计算完整权重
        weights = F.softmax(logits, dim=-1)
        
        # Top-K选择
        top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
        
        # 重新归一化Top-K权重
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        return weights, top_k_indices, top_k_weights
    
    def compute_balance_loss(self, weights: Tensor) -> Tensor:
        """
        计算负载均衡损失
        
        鼓励专家负载均匀分布，避免某些专家过度使用
        
        Args:
            weights: 专家权重 [batch, num_experts]
            
        Returns:
            balance_loss: 负载均衡损失标量
        """
        # 计算每个专家的平均负载
        expert_load = weights.mean(dim=0)  # [num_experts]
        
        # 理想的均匀分布
        uniform_load = torch.ones_like(expert_load) / self.num_experts
        
        # 计算KL散度作为不均衡度量
        balance_loss = F.kl_div(
            expert_load.log(),
            uniform_load,
            reduction='sum'
        )
        
        return balance_loss
    
    def compute_exploration_loss(self, logits: Tensor) -> Tensor:
        """
        计算探索损失
        
        鼓励路由网络探索不同的专家组合
        
        Args:
            logits: 路由logits
            
        Returns:
            exploration_loss: 探索损失
        """
        # 使用熵作为探索度量
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
        
        # 我们希望最大化熵，所以返回负熵
        return -entropy.mean()


class GatingNetwork(nn.Module):
    """
    门控网络变体 - 使用Mixture of Experts风格的门控
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        num_experts: int = 8,
        hidden_dim: int = 256,
        top_k: int = 2,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.gate = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_experts),
        )
        
    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, input_dim]
            
        Returns:
            indices: Top-K专家索引 [batch, top_k]
            weights: Top-K专家权重 [batch, top_k]
        """
        logits = self.gate(x)
        
        # Top-K gating
        top_k_logits, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(top_k_logits, dim=-1)
        
        return indices, weights


class AttentionRouter(nn.Module):
    """
    基于注意力的路由网络
    
    使用多头注意力机制来动态组合专家输出
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        num_experts: int = 8,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        # Query投影 (从输入特征)
        self.query_proj = nn.Linear(input_dim, input_dim)
        
        # Key投影 (每个专家一个可学习的key)
        self.expert_keys = nn.Parameter(torch.randn(num_experts, input_dim))
        
        # Value投影
        self.value_proj = nn.Linear(input_dim, input_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.output_proj = nn.Linear(input_dim, num_experts)
        
        self.scale = self.head_dim ** -0.5
        
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, input_dim]
            
        Returns:
            weights: 专家权重 [batch, num_experts]
        """
        batch_size = x.shape[0]
        
        # 计算query
        query = self.query_proj(x)  # [batch, input_dim]
        
        # 使用专家keys计算注意力分数
        # query: [batch, input_dim], keys: [num_experts, input_dim]
        attn_scores = torch.matmul(query, self.expert_keys.t()) * self.scale
        
        # Softmax得到权重
        weights = F.softmax(attn_scores, dim=-1)
        
        return weights


class HierarchicalRouter(nn.Module):
    """
    层次化路由网络
    
    首先进行粗粒度领域分类，然后在领域内进行细粒度专家选择
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        num_experts: int = 8,
        num_domains: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.num_experts = num_experts
        self.num_domains = num_domains
        self.experts_per_domain = num_experts // num_domains
        
        # 领域级路由
        self.domain_router = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
        )
        
        # 专家级路由 (每个领域一个)
        self.expert_routers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.experts_per_domain),
            )
            for _ in range(num_domains)
        ])
        
    def forward(self, x: Tensor) -> Tensor:
        """
        前向传播
        
        Args:
            x: 输入特征 [batch, input_dim]
            
        Returns:
            weights: 专家权重 [batch, num_experts]
        """
        batch_size = x.shape[0]
        
        # 领域级路由
        domain_logits = self.domain_router(x)
        domain_weights = F.softmax(domain_logits, dim=-1)  # [batch, num_domains]
        
        # 专家级路由
        all_expert_weights = []
        for i, router in enumerate(self.expert_routers):
            expert_logits = router(x)
            expert_weights = F.softmax(expert_logits, dim=-1)  # [batch, experts_per_domain]
            
            # 乘以领域权重
            weighted = expert_weights * domain_weights[:, i:i+1]
            all_expert_weights.append(weighted)
            
        # 合并所有专家权重
        weights = torch.cat(all_expert_weights, dim=-1)  # [batch, num_experts]
        
        return weights


if __name__ == "__main__":
    # 测试代码
    print("Testing Router Network...")
    
    batch_size = 4
    input_dim = 768
    num_experts = 8
    
    # 创建测试输入
    features = torch.randn(batch_size, input_dim)
    
    # 测试RouterNetwork
    router = RouterNetwork(
        input_dim=input_dim,
        num_experts=num_experts,
        hidden_dims=[512, 256],
        top_k=3,
    )
    
    weights, top_k_indices, top_k_weights = router(features)
    print(f"Weights shape: {weights.shape}")
    print(f"Top-K indices shape: {top_k_indices.shape}")
    print(f"Top-K weights shape: {top_k_weights.shape}")
    print(f"Weights sum: {weights.sum(dim=-1)}")
    print(f"Top-K weights sum: {top_k_weights.sum(dim=-1)}")
    
    # 测试损失计算
    balance_loss = router.compute_balance_loss(weights)
    print(f"Balance loss: {balance_loss.item():.4f}")
    
    # 测试AttentionRouter
    attn_router = AttentionRouter(input_dim, num_experts)
    attn_weights = attn_router(features)
    print(f"\nAttention router weights shape: {attn_weights.shape}")
    
    # 测试HierarchicalRouter
    hier_router = HierarchicalRouter(input_dim, num_experts)
    hier_weights = hier_router(features)
    print(f"Hierarchical router weights shape: {hier_weights.shape}")
    
    # 测试SimpleFeatureExtractor
    print("\nTesting SimpleFeatureExtractor...")
    extractor = SimpleFeatureExtractor()
    input_ids = torch.randint(0, 50000, (batch_size, 128))
    features = extractor(input_ids)
    print(f"Extracted features shape: {features.shape}")
    
    print("\nAll tests passed!")
