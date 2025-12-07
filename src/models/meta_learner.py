"""
Meta-Learner Module

Implements the final decision fusion mechanism that combines
multi-expert outputs and routing information.
"""

from typing import Optional, Tuple, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class MetaFeatureBuilder(nn.Module):
 """
 
 
 :
 1. 
 2. 
 3. 
 4. 
 """
 
 def __init__(
 self,
 num_experts: int = 8,
 expert_output_dim: int = 50000, # vocab size
 input_feature_dim: int = 768,
 routing_dim: int = 8,
 ):
 """
 Args:
 num_experts: 
 expert_output_dim: ()
 input_feature_dim: 
 routing_dim: 
 """
 super().__init__()
 
 self.num_experts = num_experts
 self.expert_output_dim = expert_output_dim
 self.input_feature_dim = input_feature_dim
 
 # ()
 self.expert_feature_dim = 16 # 
 
 # 
 self.expert_compressors = nn.ModuleList([
 nn.Sequential(
 nn.Linear(expert_output_dim, 256),
 nn.ReLU(),
 nn.Linear(256, self.expert_feature_dim),
 )
 for _ in range(num_experts)
 ])
 
 # 
 self.input_compressor = nn.Sequential(
 nn.Linear(input_feature_dim, 128),
 nn.ReLU(),
 nn.Linear(128, 64),
 )
 
 # 
 self.total_feature_dim = (
 num_experts * self.expert_feature_dim + # 
 num_experts + # 
 64 + # 
 num_experts * (num_experts - 1) // 2 + # 
 num_experts * 4 # (mean, std, max, entropy)
 )
 
 def compute_expert_statistics(
 self,
 expert_probs: List[Tensor],
 ) -> Tensor:
 """
 
 
 Args:
 expert_probs: 
 
 Returns:
 [batch, num_experts * 4]
 """
 stats_list = []
 
 for probs in expert_probs:
 # probs: [batch, seq, vocab] -> 
 if probs.dim() == 3:
 probs = probs[:, -1, :] # [batch, vocab]
 
 # 
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
 
 
 Args:
 expert_probs: 
 
 Returns:
 [batch, num_pairs]
 """
 differences = []
 
 for i in range(self.num_experts):
 for j in range(i + 1, self.num_experts):
 # 
 probs_i = expert_probs[i]
 probs_j = expert_probs[j]
 
 if probs_i.dim() == 3:
 probs_i = probs_i[:, -1, :]
 probs_j = probs_j[:, -1, :]
 
 # KL
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
 - 
 
 Args:
 expert_outputs: [batch, seq, vocab]
 routing_weights: [batch, num_experts]
 input_features: [batch, input_feature_dim]
 
 Returns:
 meta_features: [batch, total_feature_dim]
 """
 features_list = []
 
 # 1. 
 for i, (output, compressor) in enumerate(zip(expert_outputs, self.expert_compressors)):
 if output.dim() == 3:
 output = output[:, -1, :] # [batch, vocab]
 compressed = compressor(output)
 features_list.append(compressed)
 
 # 2. 
 features_list.append(routing_weights)
 
 # 3. 
 compressed_input = self.input_compressor(input_features)
 features_list.append(compressed_input)
 
 # 4. 
 pairwise_diff = self.compute_pairwise_differences(expert_outputs)
 features_list.append(pairwise_diff)
 
 # 5. 
 expert_stats = self.compute_expert_statistics(expert_outputs)
 features_list.append(expert_stats)
 
 # 
 meta_features = torch.cat(features_list, dim=-1)
 
 return meta_features

class MetaLearner(nn.Module):
 """
 
 
 。
 
 :
 p_final(x) = MetaLearner(f_meta(x))
 
 :
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
 input_dim: 
 hidden_dims: 
 activation: 
 dropout: Dropout
 output_dim: (1 for binary classification)
 """
 super().__init__()
 
 self.input_dim = input_dim
 self.output_dim = output_dim
 
 # 
 if activation == "relu":
 self.activation = nn.ReLU()
 elif activation == "gelu":
 self.activation = nn.GELU()
 elif activation == "silu":
 self.activation = nn.SiLU()
 else:
 raise ValueError(f"Unknown activation: {activation}")
 
 # MLP
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
 
 # (，logits)
 layers.append(nn.Linear(prev_dim, output_dim))
 
 self.network = nn.Sequential(*layers)
 
 # 
 self._init_weights()
 
 def _init_weights(self):
 """"""
 for module in self.modules():
 if isinstance(module, nn.Linear):
 nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
 if module.bias is not None:
 nn.init.zeros_(module.bias)
 
 def forward(self, meta_features: Tensor) -> Tensor:
 """
 
 
 Args:
 meta_features: [batch, input_dim]
 
 Returns:
 logits: logits [batch, output_dim]
 """
 return self.network(meta_features)
 
 def predict_proba(self, meta_features: Tensor) -> Tensor:
 """
 
 
 Args:
 meta_features: 
 
 Returns:
 probabilities: [batch, 1]
 """
 logits = self.forward(meta_features)
 return torch.sigmoid(logits)
 
 def predict(self, meta_features: Tensor, threshold: float = 0.5) -> Tensor:
 """
 
 
 Args:
 meta_features: 
 threshold: 
 
 Returns:
 predictions: [batch]
 """
 proba = self.predict_proba(meta_features)
 return (proba >= threshold).long().squeeze(-1)

class EnsembleDecisionMaker(nn.Module):
 """
 
 
 MetaFeatureBuilderMetaLearner，
 。
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
 num_experts: 
 expert_output_dim: 
 input_feature_dim: 
 meta_hidden_dims: 
 dropout: Dropout
 """
 super().__init__()
 
 self.num_experts = num_experts
 
 # 
 self.feature_builder = MetaFeatureBuilder(
 num_experts=num_experts,
 expert_output_dim=expert_output_dim,
 input_feature_dim=input_feature_dim,
 )
 
 # 
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
 
 
 Args:
 expert_outputs: 
 routing_weights: 
 input_features: 
 
 Returns:
 logits: logits
 """
 # 
 meta_features = self.feature_builder(
 expert_outputs, routing_weights, input_features
 )
 
 # 
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
 
 
 Args:
 expert_outputs: 
 routing_weights: 
 input_features: 
 threshold: 
 
 Returns:
 predictions: 
 probabilities: 
 """
 logits = self.forward(expert_outputs, routing_weights, input_features)
 probabilities = torch.sigmoid(logits)
 predictions = (probabilities >= threshold).long().squeeze(-1)
 
 return predictions, probabilities

class WeightedEnsemble(nn.Module):
 """
 
 
 
 """
 
 def __init__(
 self,
 num_experts: int = 8,
 combine_method: str = "weighted_sum",
 ):
 """
 Args:
 num_experts: 
 combine_method: ("weighted_sum", "attention", "gating")
 """
 super().__init__()
 
 self.num_experts = num_experts
 self.combine_method = combine_method
 
 if combine_method == "weighted_sum":
 # 
 self.weights = nn.Parameter(torch.ones(num_experts) / num_experts)
 elif combine_method == "gating":
 # 
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
 
 
 Args:
 expert_predictions: [batch, num_experts]
 routing_weights: ()
 
 Returns:
 combined: [batch, 1]
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
 feature_dim: 
 num_experts: 
 num_heads: 
 dropout: Dropout
 """
 super().__init__()
 
 self.num_experts = num_experts
 self.feature_dim = feature_dim
 
 # 
 self.attention = nn.MultiheadAttention(
 embed_dim=feature_dim,
 num_heads=num_heads,
 dropout=dropout,
 batch_first=True,
 )
 
 # 
 self.output_proj = nn.Sequential(
 nn.Linear(feature_dim, feature_dim),
 nn.ReLU(),
 nn.Linear(feature_dim, feature_dim),
 )
 
 def forward(self, expert_features: Tensor) -> Tensor:
 """
 
 
 Args:
 expert_features: [batch, num_experts, feature_dim]
 
 Returns:
 fused: [batch, feature_dim]
 """
 # 
 attn_output, _ = self.attention(
 expert_features, expert_features, expert_features
 )
 
 # 
 attn_output = attn_output + expert_features
 
 # 
 pooled = attn_output.mean(dim=1)
 
 # 
 output = self.output_proj(pooled)
 
 return output

if __name__ == "__main__":
 # 
 print("Testing Meta Learner Module...")
 
 batch_size = 4
 num_experts = 8
 vocab_size = 50000
 input_feature_dim = 768
 
 # 
 expert_outputs = [
 torch.randn(batch_size, vocab_size).softmax(dim=-1)
 for _ in range(num_experts)
 ]
 routing_weights = torch.randn(batch_size, num_experts).softmax(dim=-1)
 input_features = torch.randn(batch_size, input_feature_dim)
 
 # MetaFeatureBuilder
 print("Testing MetaFeatureBuilder...")
 feature_builder = MetaFeatureBuilder(
 num_experts=num_experts,
 expert_output_dim=vocab_size,
 input_feature_dim=input_feature_dim,
 )
 meta_features = feature_builder(expert_outputs, routing_weights, input_features)
 print(f"Meta features shape: {meta_features.shape}")
 print(f"Total feature dim: {feature_builder.total_feature_dim}")
 
 # MetaLearner
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
 
 # EnsembleDecisionMaker
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
 
 # WeightedEnsemble
 print("\nTesting WeightedEnsemble...")
 weighted_ensemble = WeightedEnsemble(num_experts)
 expert_preds = torch.randn(batch_size, num_experts).sigmoid()
 combined = weighted_ensemble(expert_preds)
 print(f"Combined prediction shape: {combined.shape}")
 
 # AttentionFusion
 print("\nTesting AttentionFusion...")
 fusion = AttentionFusion(feature_dim=256, num_experts=num_experts)
 expert_features = torch.randn(batch_size, num_experts, 256)
 fused = fusion(expert_features)
 print(f"Fused features shape: {fused.shape}")
 
 print("\nAll tests passed!")
