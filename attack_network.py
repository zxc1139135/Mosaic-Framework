"""
Contrastive attack network for membership inference.
  1. FeatureEncoder: 3-layer MLP -> l2-normalized embedding
  2. SupConLoss: supervised contrastive loss
  3. ClassificationHead: linear -> sigmoid membership prob
  4. AttackLoss: L_attack = L_con + lambda * L_cls
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class FeatureEncoder(nn.Module):
    """3-layer MLP encoder: R^{1+5K} -> R^d (l2-normalized)."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=-1)


class ClassificationHead(nn.Module):
    """Linear head: sigmoid(u^T e + b) -> membership probability."""

    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.linear = nn.Linear(embed_dim, 1)

    def forward(self, e):
        return torch.sigmoid(self.linear(e)).squeeze(-1)


class ContrastiveAttackNetwork(nn.Module):
    """Complete attack network: encoder + classifier."""

    def __init__(self, input_dim, hidden_dim=256, embed_dim=128):
        super().__init__()
        self.encoder = FeatureEncoder(input_dim, hidden_dim, embed_dim)
        self.classifier = ClassificationHead(embed_dim)

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (embeddings, membership_probs)."""
        e = self.encoder(x)
        p = self.classifier(e)
        return e, p

    def predict(self, x, threshold=0.5):
        _, p = self.forward(x)
        return (p >= threshold).long()

    def predict_proba(self, x):
        _, p = self.forward(x)
        return p


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Eq. 8).
    Pulls same-label embeddings together, pushes different-label apart.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, embeddings, labels):
        device = embeddings.device
        B = embeddings.size(0)
        if B <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Cosine similarity (embeddings already l2-normalized)
        sim = torch.matmul(embeddings, embeddings.T) / self.temperature

        # Positive mask: same label, exclude self
        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        self_mask = torch.eye(B, device=device)
        pos_mask = pos_mask - self_mask

        # Numerical stability
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        # Log-softmax over negatives
        exp_sim = torch.exp(sim) * (1 - self_mask)
        log_denom = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        log_prob = sim - log_denom

        # Average over positives per anchor
        num_pos = pos_mask.sum(dim=1)
        valid = num_pos > 0
        if valid.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / num_pos.clamp(min=1)
        return -mean_log_prob[valid].mean()


class AttackLoss(nn.Module):
    """Joint loss: L_attack = L_con + lambda * L_cls (Eq. 10)."""

    def __init__(self, temperature=0.07, lambda_cls=0.5):
        super().__init__()
        self.supcon = SupConLoss(temperature)
        self.bce = nn.BCELoss()
        self.lambda_cls = lambda_cls

    def forward(self, embeddings, probs, labels):
        l_con = self.supcon(embeddings, labels)
        l_cls = self.bce(probs, labels.float())
        return l_con + self.lambda_cls * l_cls, l_con, l_cls
