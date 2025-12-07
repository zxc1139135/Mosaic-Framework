"""
Router Network Module

Implements dynamic routing mechanism for intelligent expert selection
based on input features.
"""

import math
from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class FeatureExtractor(nn.Module):
 """
 - BERT
 
 (4): φ(x) = MeanPooling(BERT(x)) ∈ R^768
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
 model_name: 
 output_dim: 
 pooling: ("mean", "cls", "max")
 freeze: BERT
 """
 super().__init__()
 
 self.output_dim = output_dim
 self.pooling = pooling
 
 # BERT (import)
 self._bert = None
 self._tokenizer = None
 self.model_name = model_name
 self.freeze = freeze
 
 def _load_bert(self):
 """BERT"""
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
 
 
 Args:
 input_ids: Token IDs [batch, seq]
 attention_mask: 
 text: (，tokenize)
 
 Returns:
 [batch, output_dim]
 """
 # ，tokenization
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
 
 # 
 device = next(self.bert.parameters()).device
 input_ids = input_ids.to(device)
 attention_mask = attention_mask.to(device)
 
 # BERT
 outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
 hidden_states = outputs.last_hidden_state # [batch, seq, hidden]
 
 # 
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
 - BERT，
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
 """"""
 embedded = self.embedding(input_ids)
 outputs, (h_n, _) = self.lstm(embedded)
 
 # 
 features = torch.cat([h_n[-2], h_n[-1]], dim=-1)
 return features

class RouterNetwork(nn.Module):
 """
 - 
 
 ，K。
 
 :
 - : {w_k(x)}_{k=1}^8 = RouteNet(φ(x))
 - Top-K
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
 input_dim: 
 num_experts: 
 hidden_dims: 
 activation: 
 dropout: Dropout
 top_k: Top-K
 temperature: Softmax
 """
 super().__init__()
 
 self.num_experts = num_experts
 self.top_k = top_k
 self.temperature = temperature
 
 # 
 if activation == "gelu":
 self.activation = nn.GELU()
 elif activation == "relu":
 self.activation = nn.ReLU()
 elif activation == "silu":
 self.activation = nn.SiLU()
 else:
 raise ValueError(f"Unknown activation: {activation}")
 
 # MLP
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
 
 # 
 layers.append(nn.Linear(prev_dim, num_experts))
 
 self.router = nn.Sequential(*layers)
 
 # ()
 self.noise_std = 0.1
 
 def forward(
 self,
 features: Tensor,
 add_noise: bool = False,
 ) -> Tuple[Tensor, Tensor, Tensor]:
 """
 
 
 Args:
 features: [batch, input_dim]
 add_noise: ()
 
 Returns:
 weights: [batch, num_experts]
 top_k_indices: Top-K [batch, top_k]
 top_k_weights: Top-K [batch, top_k]
 """
 # logits
 logits = self.router(features) # [batch, num_experts]
 
 # ：
 if add_noise and self.training:
 noise = torch.randn_like(logits) * self.noise_std
 logits = logits + noise
 
 # 
 logits = logits / self.temperature
 
 # 
 weights = F.softmax(logits, dim=-1)
 
 # Top-K
 top_k_weights, top_k_indices = torch.topk(weights, self.top_k, dim=-1)
 
 # Top-K
 top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
 
 return weights, top_k_indices, top_k_weights
 
 def compute_balance_loss(self, weights: Tensor) -> Tensor:
 """
 
 
 ，
 
 Args:
 weights: [batch, num_experts]
 
 Returns:
 balance_loss: 
 """
 # 
 expert_load = weights.mean(dim=0) # [num_experts]
 
 # 
 uniform_load = torch.ones_like(expert_load) / self.num_experts
 
 # KL
 balance_loss = F.kl_div(
 expert_load.log(),
 uniform_load,
 reduction='sum'
 )
 
 return balance_loss
 
 def compute_exploration_loss(self, logits: Tensor) -> Tensor:
 """
 
 
 
 
 Args:
 logits: logits
 
 Returns:
 exploration_loss: 
 """
 # 
 probs = F.softmax(logits, dim=-1)
 entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
 
 # ，
 return -entropy.mean()

class GatingNetwork(nn.Module):
 """
 - Mixture of Experts
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
 
 
 Args:
 x: [batch, input_dim]
 
 Returns:
 indices: Top-K [batch, top_k]
 weights: Top-K [batch, top_k]
 """
 logits = self.gate(x)
 
 # Top-K gating
 top_k_logits, indices = torch.topk(logits, self.top_k, dim=-1)
 weights = F.softmax(top_k_logits, dim=-1)
 
 return indices, weights

class AttentionRouter(nn.Module):
 """
 
 
 
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
 
 # Query ()
 self.query_proj = nn.Linear(input_dim, input_dim)
 
 # Key (key)
 self.expert_keys = nn.Parameter(torch.randn(num_experts, input_dim))
 
 # Value
 self.value_proj = nn.Linear(input_dim, input_dim)
 
 self.dropout = nn.Dropout(dropout)
 self.output_proj = nn.Linear(input_dim, num_experts)
 
 self.scale = self.head_dim ** -0.5
 
 def forward(self, x: Tensor) -> Tensor:
 """
 
 
 Args:
 x: [batch, input_dim]
 
 Returns:
 weights: [batch, num_experts]
 """
 batch_size = x.shape[0]
 
 # query
 query = self.query_proj(x) # [batch, input_dim]
 
 # keys
 # query: [batch, input_dim], keys: [num_experts, input_dim]
 attn_scores = torch.matmul(query, self.expert_keys.t()) * self.scale
 
 # Softmax
 weights = F.softmax(attn_scores, dim=-1)
 
 return weights

class HierarchicalRouter(nn.Module):
 """
 
 
 ，
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
 
 # 
 self.domain_router = nn.Sequential(
 nn.Linear(input_dim, hidden_dim),
 nn.ReLU(),
 nn.Linear(hidden_dim, num_domains),
 )
 
 # ()
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
 
 
 Args:
 x: [batch, input_dim]
 
 Returns:
 weights: [batch, num_experts]
 """
 batch_size = x.shape[0]
 
 # 
 domain_logits = self.domain_router(x)
 domain_weights = F.softmax(domain_logits, dim=-1) # [batch, num_domains]
 
 # 
 all_expert_weights = []
 for i, router in enumerate(self.expert_routers):
 expert_logits = router(x)
 expert_weights = F.softmax(expert_logits, dim=-1) # [batch, experts_per_domain]
 
 # 
 weighted = expert_weights * domain_weights[:, i:i+1]
 all_expert_weights.append(weighted)
 
 # 
 weights = torch.cat(all_expert_weights, dim=-1) # [batch, num_experts]
 
 return weights

if __name__ == "__main__":
 # 
 print("Testing Router Network...")
 
 batch_size = 4
 input_dim = 768
 num_experts = 8
 
 # 
 features = torch.randn(batch_size, input_dim)
 
 # RouterNetwork
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
 
 # 
 balance_loss = router.compute_balance_loss(weights)
 print(f"Balance loss: {balance_loss.item():.4f}")
 
 # AttentionRouter
 attn_router = AttentionRouter(input_dim, num_experts)
 attn_weights = attn_router(features)
 print(f"\nAttention router weights shape: {attn_weights.shape}")
 
 # HierarchicalRouter
 hier_router = HierarchicalRouter(input_dim, num_experts)
 hier_weights = hier_router(features)
 print(f"Hierarchical router weights shape: {hier_weights.shape}")
 
 # SimpleFeatureExtractor
 print("\nTesting SimpleFeatureExtractor...")
 extractor = SimpleFeatureExtractor()
 input_ids = torch.randint(0, 50000, (batch_size, 128))
 features = extractor(input_ids)
 print(f"Extracted features shape: {features.shape}")
 
 print("\nAll tests passed!")
