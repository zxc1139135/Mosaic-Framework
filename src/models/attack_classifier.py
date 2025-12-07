"""
Attack Classifier Module

Implements the membership inference attack classifier that processes
extracted features to determine training set membership.
"""

from typing import Optional, Tuple, Dict, List, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class AttackFeatureExtractor(nn.Module):
    """
    Feature extractor for membership inference attacks.
    
    Extracts comprehensive features from model outputs for attack classification.
    
    Feature dimensions:
        - Target model features: 12
        - Multi-expert features: 28
        - Interaction features: 5
        - Total: 45
    """
    
    def __init__(
        self,
        target_feature_dim: int = 12,
        multi_expert_feature_dim: int = 28,
        interaction_feature_dim: int = 5,
        num_experts: int = 8,
    ):
        """
        Initialize the feature extractor.
        
        Args:
            target_feature_dim: Dimension of target model features
            multi_expert_feature_dim: Dimension of multi-expert features
            interaction_feature_dim: Dimension of interaction features
            num_experts: Number of expert models
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
        Extract features from target model outputs (12 dimensions).
        
        Features include:
            1. Loss value (1 dim)
            2. Perplexity (1 dim)
            3. Top-1 probability statistics (3 dims: mean, std, max)
            4. Entropy statistics (3 dims: mean, std, min)
            5. Token-level features (4 dims: first, last, mean, min)
        
        Args:
            target_probs: Probability distribution [batch, seq, vocab]
            target_loss: Loss values [batch]
            input_ids: Token IDs [batch, seq]
        
        Returns:
            target_features: Feature tensor [batch, 12]
        """
        batch_size = target_probs.shape[0]
        features = []
        
        # Loss value
        features.append(target_loss.unsqueeze(-1))
        
        # Perplexity
        perplexity = torch.exp(target_loss).unsqueeze(-1)
        features.append(perplexity)
        
        # Top-1 probability statistics
        top1_probs = target_probs.max(dim=-1)[0]
        features.append(top1_probs.mean(dim=-1, keepdim=True))
        features.append(top1_probs.std(dim=-1, keepdim=True))
        features.append(top1_probs.max(dim=-1, keepdim=True)[0])
        
        # Entropy statistics
        entropy = -(target_probs * torch.log(target_probs + 1e-10)).sum(dim=-1)
        features.append(entropy.mean(dim=-1, keepdim=True))
        features.append(entropy.std(dim=-1, keepdim=True))
        features.append(entropy.min(dim=-1, keepdim=True)[0])
        
        # Token-level probability features
        seq_length = target_probs.shape[1]
        if seq_length > 1:
            # Shift for causal language modeling
            shifted_probs = target_probs[:, :-1, :]
            shifted_ids = input_ids[:, 1:]
            
            # Get probability of correct tokens
            correct_probs = shifted_probs.gather(
                dim=-1, 
                index=shifted_ids.unsqueeze(-1)
            ).squeeze(-1)
            
            features.append(correct_probs[:, 0:1])  # First token
            features.append(correct_probs[:, -1:])  # Last token
            features.append(correct_probs.mean(dim=-1, keepdim=True))
            features.append(correct_probs.min(dim=-1, keepdim=True)[0])
        else:
            # Padding for sequences of length 1
            features.extend([torch.zeros(batch_size, 1, device=target_probs.device)] * 4)
        
        return torch.cat(features, dim=-1)
    
    def extract_multi_expert_features(
        self,
        expert_probs_list: List[Tensor],
        routing_weights: Tensor,
    ) -> Tensor:
        """
        Extract features from multi-expert system (28 dimensions).
        
        Features include:
            1. Per-expert top-1 probability means (8 dims)
            2. Per-expert entropy means (8 dims)
            3. Routing weights (8 dims)
            4. Aggregated statistics (4 dims)
        
        Args:
            expert_probs_list: List of probability tensors [batch, seq, vocab] x num_experts
            routing_weights: Expert selection weights [batch, num_experts]
        
        Returns:
            expert_features: Feature tensor [batch, 28]
        """
        features = []
        
        expert_top1_means = []
        expert_entropy_means = []
        
        for expert_probs in expert_probs_list:
            # Top-1 probability mean
            top1_probs = expert_probs.max(dim=-1)[0]
            top1_mean = top1_probs.mean(dim=-1, keepdim=True)
            expert_top1_means.append(top1_mean)
            
            # Entropy mean
            entropy = -(expert_probs * torch.log(expert_probs + 1e-10)).sum(dim=-1)
            entropy_mean = entropy.mean(dim=-1, keepdim=True)
            expert_entropy_means.append(entropy_mean)
        
        # Per-expert top-1 means
        features.extend(expert_top1_means)
        
        # Per-expert entropy means
        features.extend(expert_entropy_means)
        
        # Routing weights
        features.append(routing_weights)
        
        # Aggregated statistics
        stacked_top1 = torch.cat(expert_top1_means, dim=-1)
        
        # Expert variance
        expert_variance = stacked_top1.var(dim=-1, keepdim=True)
        features.append(expert_variance)
        
        # Top-k weight concentration
        top3_weights = routing_weights.topk(3, dim=-1)[0].sum(dim=-1, keepdim=True)
        features.append(top3_weights)
        
        # Expert range
        expert_range = stacked_top1.max(dim=-1, keepdim=True)[0] - stacked_top1.min(dim=-1, keepdim=True)[0]
        features.append(expert_range)
        
        # Weighted entropy
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
        Extract interaction features between target and experts (5 dimensions).
        
        Features include:
            1. Forward KL divergence
            2. Reverse KL divergence
            3. Cosine similarity with best expert
            4. Mean absolute difference
            5. Difference standard deviation
        
        Args:
            target_probs: Target model probabilities [batch, seq, vocab]
            expert_probs_list: List of expert probability tensors
            routing_weights: Expert routing weights
        
        Returns:
            interaction_features: Feature tensor [batch, 5]
        """
        features = []
        
        # Compute weighted expert probabilities
        weighted_expert_probs = torch.zeros_like(target_probs)
        for i, expert_probs in enumerate(expert_probs_list):
            weighted_expert_probs += routing_weights[:, i:i+1, None] * expert_probs
        
        # Forward KL divergence (experts || target)
        kl_div = F.kl_div(
            weighted_expert_probs.log(),
            target_probs,
            reduction='none'
        ).sum(dim=-1).mean(dim=-1, keepdim=True)
        features.append(kl_div)
        
        # Reverse KL divergence (target || experts)
        reverse_kl = F.kl_div(
            target_probs.log(),
            weighted_expert_probs,
            reduction='none'
        ).sum(dim=-1).mean(dim=-1, keepdim=True)
        features.append(reverse_kl)
        
        # Cosine similarity with best expert
        best_expert_idx = routing_weights.argmax(dim=-1)
        batch_size = target_probs.shape[0]
        
        target_last = target_probs[:, -1, :]
        best_expert_probs = torch.stack([
            expert_probs_list[best_expert_idx[i].item()][i, -1, :]
            for i in range(batch_size)
        ])
        
        cosine_sim = F.cosine_similarity(target_last, best_expert_probs, dim=-1)
        features.append(cosine_sim.unsqueeze(-1))
        
        # Mean absolute difference
        diff = (target_probs - weighted_expert_probs).abs().mean(dim=(-1, -2), keepdim=True)
        features.append(diff.squeeze(-1))
        
        # Difference standard deviation
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
        Extract all features for attack classification.
        
        Args:
            target_probs: Target model probability distribution
            target_loss: Target model loss
            input_ids: Input token IDs
            expert_probs_list: List of expert probability distributions
            routing_weights: Expert routing weights
        
        Returns:
            attack_features: Combined feature tensor [batch, 45]
        """
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
    """
    Multi-layer perceptron classifier for membership inference.
    
    Architecture (from paper equations 6-8):
        h1 = ReLU(W1 * f_attack(x) + b1), W1 in R^{128x45}
        h2 = ReLU(W2 * h1 + b2), W2 in R^{64x128}
        p_attack(x) = sigmoid(W3 * h2 + b3), W3 in R^{1x64}
    """
    
    def __init__(
        self,
        input_dim: int = 45,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.3,
    ):
        """
        Initialize the attack classifier.
        
        Args:
            input_dim: Input feature dimension (default: 45)
            hidden_dims: Hidden layer dimensions (default: [128, 64])
            dropout: Dropout probability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        
        # First hidden layer: 45 -> 128
        self.layer1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Second hidden layer: 128 -> 64
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Output layer: 64 -> 1
        self.output_layer = nn.Linear(hidden_dims[1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the classifier.
        
        Args:
            x: Input features [batch, input_dim]
        
        Returns:
            logits: Output logits [batch, 1]
        """
        h1 = self.layer1(x)
        h2 = self.layer2(h1)
        logits = self.output_layer(h2)
        return logits
    
    def predict_proba(self, x: Tensor) -> Tensor:
        """Get membership probability predictions."""
        logits = self.forward(x)
        return torch.sigmoid(logits)
    
    def predict(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        """Get binary membership predictions."""
        proba = self.predict_proba(x)
        return (proba >= threshold).long().squeeze(-1)


class MembershipInferenceAttack(nn.Module):
    """
    Complete membership inference attack pipeline.
    
    Combines feature extraction and classification to determine
    whether a sample was part of the target model's training set.
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
        Initialize the membership inference attack.
        
        Args:
            num_experts: Number of expert models
            target_feature_dim: Target model feature dimension
            multi_expert_feature_dim: Multi-expert feature dimension
            interaction_feature_dim: Interaction feature dimension
            classifier_hidden_dims: Classifier hidden layer dimensions
            dropout: Dropout probability
        """
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
        
        # Feature normalization parameters
        self.register_buffer('feature_mean', None)
        self.register_buffer('feature_std', None)
    
    def fit_scaler(self, features: Tensor):
        """
        Fit feature normalization parameters.
        
        From paper Section 5.2: f'_attack(x) = (f_attack(x) - mu) / sigma
        
        Args:
            features: Training features [N, feature_dim]
        """
        self.feature_mean = features.mean(dim=0)
        self.feature_std = features.std(dim=0).clamp(min=1e-6)
    
    def normalize_features(self, features: Tensor) -> Tensor:
        """Apply Z-score normalization to features."""
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
        """Extract and normalize attack features."""
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
        Forward pass for membership inference.
        
        Args:
            target_probs: Target model probability distribution
            target_loss: Target model loss
            input_ids: Input token IDs
            expert_probs_list: List of expert probability distributions
            routing_weights: Expert routing weights
        
        Returns:
            logits: Membership inference logits
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
        Perform membership inference attack.
        
        Args:
            target_probs: Target model probability distribution
            target_loss: Target model loss
            input_ids: Input token IDs
            expert_probs_list: Expert probability distributions
            routing_weights: Expert routing weights
            threshold: Classification threshold
        
        Returns:
            predictions: Binary membership predictions [batch]
            probabilities: Membership probabilities [batch, 1]
        """
        logits = self.forward(
            target_probs, target_loss, input_ids,
            expert_probs_list, routing_weights
        )
        probabilities = torch.sigmoid(logits)
        predictions = (probabilities >= threshold).long().squeeze(-1)
        
        return predictions, probabilities


if __name__ == "__main__":
    print("Testing Attack Classifier Module...")
    
    batch_size = 8
    seq_length = 128
    vocab_size = 50000
    num_experts = 8
    
    # Create test data
    target_probs = torch.randn(batch_size, seq_length, vocab_size).softmax(dim=-1)
    target_loss = torch.rand(batch_size)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
    expert_probs_list = [
        torch.randn(batch_size, seq_length, vocab_size).softmax(dim=-1)
        for _ in range(num_experts)
    ]
    routing_weights = torch.randn(batch_size, num_experts).softmax(dim=-1)
    
    # Test feature extractor
    print("Testing AttackFeatureExtractor...")
    feature_extractor = AttackFeatureExtractor(num_experts=num_experts)
    features = feature_extractor(
        target_probs, target_loss, input_ids,
        expert_probs_list, routing_weights
    )
    print(f"Feature shape: {features.shape}")
    print(f"Expected dim: {feature_extractor.total_feature_dim}")
    
    # Test classifier
    print("\nTesting AttackClassifier...")
    classifier = AttackClassifier(input_dim=features.shape[-1])
    logits = classifier(features)
    print(f"Logits shape: {logits.shape}")
    
    predictions = classifier.predict(features)
    print(f"Predictions shape: {predictions.shape}")
    
    # Test complete attack pipeline
    print("\nTesting MembershipInferenceAttack...")
    attack = MembershipInferenceAttack(num_experts=num_experts)
    attack.fit_scaler(features)
    
    preds, probs = attack.attack(
        target_probs, target_loss, input_ids,
        expert_probs_list, routing_weights
    )
    print(f"Attack predictions shape: {preds.shape}")
    print(f"Attack probabilities shape: {probs.shape}")
    
    print("\nAll tests passed!")
