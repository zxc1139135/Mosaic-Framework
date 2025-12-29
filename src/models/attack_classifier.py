"""
Attack Classifier Module
========================
"""

from typing import Optional, Tuple, Dict, List, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AttackFeatureExtractor(nn.Module):
    
    def __init__(
        self,
        target_feature_dim: int = 12,
        multi_expert_feature_dim: int = 28,
        interaction_feature_dim: int = 5,
        num_experts: int = 8,
    ):

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

        batch_size = target_probs.shape[0]
        features = []

        features.append(target_loss.unsqueeze(-1))  # [batch, 1]

        perplexity = torch.exp(target_loss).unsqueeze(-1)
        features.append(perplexity)  # [batch, 1]

        top1_probs = target_probs.max(dim=-1)[0]  # [batch, seq]
        features.append(top1_probs.mean(dim=-1, keepdim=True))
        features.append(top1_probs.std(dim=-1, keepdim=True))
        features.append(top1_probs.max(dim=-1, keepdim=True)[0])

        entropy = -(target_probs * torch.log(target_probs + 1e-10)).sum(dim=-1)  # [batch, seq]
        features.append(entropy.mean(dim=-1, keepdim=True))
        features.append(entropy.std(dim=-1, keepdim=True))
        features.append(entropy.min(dim=-1, keepdim=True)[0])

        seq_length = target_probs.shape[1]
        if seq_length > 1:
            # Shift for causal LM
            shifted_probs = target_probs[:, :-1, :]  # [batch, seq-1, vocab]
            shifted_ids = input_ids[:, 1:]  # [batch, seq-1]

            correct_probs = shifted_probs.gather(
                dim=-1, 
                index=shifted_ids.unsqueeze(-1)
            ).squeeze(-1)  # [batch, seq-1]
            
            features.append(correct_probs[:, 0:1])
            features.append(correct_probs[:, -1:])
            features.append(correct_probs.mean(dim=-1, keepdim=True))
            features.append(correct_probs.min(dim=-1, keepdim=True)[0])
        else:
            features.extend([torch.zeros(batch_size, 1, device=target_probs.device)] * 4)
        
        return torch.cat(features, dim=-1)
    
    def extract_multi_expert_features(
        self,
        expert_probs_list: List[Tensor],
        routing_weights: Tensor,
    ) -> Tensor:

        features = []
        
        expert_top1_means = []
        expert_entropy_means = []
        
        for expert_probs in expert_probs_list:
            top1_probs = expert_probs.max(dim=-1)[0]  # [batch, seq]
            top1_mean = top1_probs.mean(dim=-1, keepdim=True)
            expert_top1_means.append(top1_mean)

            entropy = -(expert_probs * torch.log(expert_probs + 1e-10)).sum(dim=-1)
            entropy_mean = entropy.mean(dim=-1, keepdim=True)
            expert_entropy_means.append(entropy_mean)

        features.extend(expert_top1_means)

        features.extend(expert_entropy_means)

        features.append(routing_weights)

        stacked_top1 = torch.cat(expert_top1_means, dim=-1)  # [batch, num_experts]

        expert_variance = stacked_top1.var(dim=-1, keepdim=True)
        features.append(expert_variance)

        top3_weights = routing_weights.topk(3, dim=-1)[0].sum(dim=-1, keepdim=True)
        features.append(top3_weights)

        expert_range = stacked_top1.max(dim=-1, keepdim=True)[0] - stacked_top1.min(dim=-1, keepdim=True)[0]
        features.append(expert_range)

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
        features = []

        weighted_expert_probs = torch.zeros_like(target_probs)
        for i, expert_probs in enumerate(expert_probs_list):
            weighted_expert_probs += routing_weights[:, i:i+1, None] * expert_probs

        kl_div = F.kl_div(
            weighted_expert_probs.log(),
            target_probs,
            reduction='none'
        ).sum(dim=-1).mean(dim=-1, keepdim=True)
        features.append(kl_div)

        reverse_kl = F.kl_div(
            target_probs.log(),
            weighted_expert_probs,
            reduction='none'
        ).sum(dim=-1).mean(dim=-1, keepdim=True)
        features.append(reverse_kl)

        best_expert_idx = routing_weights.argmax(dim=-1)  # [batch]
        batch_size = target_probs.shape[0]

        target_last = target_probs[:, -1, :]  # [batch, vocab]
        best_expert_probs = torch.stack([
            expert_probs_list[best_expert_idx[i].item()][i, -1, :]
            for i in range(batch_size)
        ])  # [batch, vocab]

        cosine_sim = F.cosine_similarity(target_last, best_expert_probs, dim=-1)
        features.append(cosine_sim.unsqueeze(-1))

        diff = (target_probs - weighted_expert_probs).abs().mean(dim=(-1, -2), keepdim=True)
        features.append(diff.squeeze(-1))

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
        target_features = self.extract_target_features(
            target_probs, target_loss, input_ids
        )
        
        multi_expert_features = self.extract_multi_expert_features(
            expert_probs_list, routing_weights
        )
        
        interaction_features = self.extract_interaction_features(
            target_probs, expert_probs_list, routing_weights
        )

        attack_features = torch.cat([
            target_features,
            multi_expert_features,
            interaction_features,
        ], dim=-1)
        
        return attack_features


class AttackClassifier(nn.Module):
    
    def __init__(
        self,
        input_dim: int = 45,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.3,
    ):

        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output_layer = nn.Linear(hidden_dims[1], 1)

        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x: Tensor) -> Tensor:
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        logits = self.output_layer(h2)
        return logits
    
    def predict_proba(self, x: Tensor) -> Tensor:
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def predict(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        proba = self.predict_proba(x)
        return (proba >= threshold).long().squeeze(-1)


class MembershipInferenceAttack(nn.Module):
    
    def __init__(
        self,
        num_experts: int = 8,
        target_feature_dim: int = 12,
        multi_expert_feature_dim: int = 28,
        interaction_feature_dim: int = 5,
        classifier_hidden_dims: List[int] = [128, 64],
        dropout: float = 0.3,
    ):
        super().__init__()
        self.feature_extractor = AttackFeatureExtractor(
            target_feature_dim=target_feature_dim,
            multi_expert_feature_dim=multi_expert_feature_dim,
            interaction_feature_dim=interaction_feature_dim,
            num_experts=num_experts,
        )

        self.classifier = AttackClassifier(
            input_dim=self.feature_extractor.total_feature_dim,
            hidden_dims=classifier_hidden_dims,
            dropout=dropout,
        )

        self.register_buffer('feature_mean', None)
        self.register_buffer('feature_std', None)
        
    def fit_scaler(self, features: Tensor):
        self.feature_mean = features.mean(dim=0)
        self.feature_std = features.std(dim=0).clamp(min=1e-6)
        
    def normalize_features(self, features: Tensor) -> Tensor:
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

        logits = self.forward(
            target_probs, target_loss, input_ids,
            expert_probs_list, routing_weights
        )
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).long().squeeze(-1)
        
        return predictions, probabilities


class BaselineAttacks:
    @staticmethod
    def loss_based_attack(
        loss: Tensor,
        threshold: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:

        scores = -loss
        
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
        scores = reference_loss - target_loss
        predictions = (scores >= threshold).long()
        return predictions, scores
    
    @staticmethod
    def min_k_attack(
        probs: Tensor,
        k_ratio: float = 0.2,
        threshold: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:
        k = max(1, int(probs.shape[1] * k_ratio))
        min_k_probs = probs.topk(k, dim=1, largest=False)[0]  # [batch, k]
        scores = min_k_probs.mean(dim=1)
        
        if threshold is None:
            threshold = scores.median()

        predictions = (scores >= threshold).long()
        return predictions, scores
    
    @staticmethod
    def zlib_attack(
        loss: Tensor,
        zlib_entropy: Tensor,
        threshold: Optional[float] = None,
    ) -> Tuple[Tensor, Tensor]:

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

        mean_perturbed_loss = perturbed_losses.mean(dim=1)
        scores = mean_perturbed_loss - original_loss
        
        if threshold is None:
            threshold = scores.median()

        predictions = (scores >= threshold).long()
        return predictions, scores


if __name__ == "__main__":
    print("Testing Attack Classifier Module...")
    
    batch_size = 8
    seq_length = 128
    vocab_size = 50000
    num_experts = 8

    target_probs = torch.randn(batch_size, seq_length, vocab_size).softmax(dim=-1)
    target_loss = torch.rand(batch_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    expert_probs_list = [
        torch.randn(batch_size, seq_length, vocab_size).softmax(dim=-1)
        for _ in range(num_experts)
    ]
    routing_weights = torch.randn(batch_size, num_experts).softmax(dim=-1)

    print("Testing AttackFeatureExtractor...")
    feature_extractor = AttackFeatureExtractor(num_experts=num_experts)
    features = feature_extractor(
        target_probs, target_loss, input_ids,
        expert_probs_list, routing_weights
    )
    print(f"Feature shape: {features.shape}")
    print(f"Expected dim: {feature_extractor.total_feature_dim}")

    print("\nTesting AttackClassifier...")
    classifier = AttackClassifier(input_dim=features.shape[-1])
    logits = classifier(features)
    print(f"Logits shape: {logits.shape}")
    
    predictions = classifier.predict(features)
    print(f"Predictions shape: {predictions.shape}")
    print(f"Predictions: {predictions}")

    print("\nTesting MembershipInferenceAttack...")
    attack = MembershipInferenceAttack(num_experts=num_experts)

    attack.fit_scaler(features)
    
    preds, probs = attack.attack(
        target_probs, target_loss, input_ids,
        expert_probs_list, routing_weights
    )
    print(f"Attack predictions shape: {preds.shape}")
    print(f"Attack probabilities shape: {probs.shape}")

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
