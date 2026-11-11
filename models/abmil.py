"""
MIL models: MeanPool, ABMIL (attention-based MIL), and GABMIL (gated attention).
All take (N, MAX_PAD, n_feat) + padding_mask and produce scalar logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeanPool(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_masked = x * mask.unsqueeze(-1).float()
        bag_rep  = x_masked.sum(dim=1) / mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        return self.net(bag_rep).squeeze(-1)


class ABMIL(nn.Module):
    def __init__(self, n_feat: int, hidden: int = 64, attn_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.feat_net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.attn_V = nn.Linear(hidden, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.feat_net(x)                          # (B, L, hidden)
        a = torch.tanh(self.attn_V(h))                # (B, L, attn_dim)
        a = self.attn_w(a).squeeze(-1)                # (B, L)
        a = a.masked_fill(~mask, float("-inf"))
        a = F.softmax(a, dim=-1)                      # (B, L)
        bag = (a.unsqueeze(-1) * h).sum(dim=1)        # (B, hidden)
        return self.classifier(bag).squeeze(-1)


class GABMIL(nn.Module):
    """Gated attention MIL."""
    def __init__(self, n_feat: int, hidden: int = 64, attn_dim: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.feat_net = nn.Sequential(
            nn.Linear(n_feat, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.attn_V = nn.Linear(hidden, attn_dim)
        self.attn_U = nn.Linear(hidden, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.feat_net(x)
        a = torch.tanh(self.attn_V(h)) * torch.sigmoid(self.attn_U(h))
        a = self.attn_w(a).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf"))
        a = F.softmax(a, dim=-1)
        bag = (a.unsqueeze(-1) * h).sum(dim=1)
        return self.classifier(bag).squeeze(-1)
