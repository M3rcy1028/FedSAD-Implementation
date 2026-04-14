import torch
import torch.nn as nn


class FusionBF(nn.Module):
    """Bilinear fusion."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.bilinear = nn.Bilinear(dim, dim, dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, g_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        out = self.bilinear(g_feat, s_feat)
        out = self.activation(out)
        return self.dropout(out)


class FusionSAF(nn.Module):
    """Self-attention fusion."""

    def __init__(self, dim, nhead=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, g_feat, s_feat):
        x = torch.stack([g_feat, s_feat], dim=1)  # (B,2,dim)
        attn_out, _ = self.attn(x, x, x)
        x = self.norm(x + attn_out)
        out = self.fc(x.mean(dim=1))
        return self.drop(out)


class FusionHIF(nn.Module):
    """Higher-order interaction fusion."""

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        input_dim = dim * 5  # [g, s, g^2, s^2, g*s]
        self.linear = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, dim),
        )

    def forward(self, g_feat: torch.Tensor, s_feat: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            [
                g_feat,
                s_feat,
                torch.square(g_feat),
                torch.square(s_feat),
                g_feat * s_feat,
            ],
            dim=-1,
        )
        return self.linear(features)


def build_fusion(fusion_type: str, dim: int, nhead: int = 4, dropout: float = 0.1) -> nn.Module:
    fusion_type = fusion_type.lower()
    if fusion_type == "bf":
        return FusionBF(dim, dropout=dropout)
    if fusion_type == "saf":
        return FusionSAF(dim, nhead=nhead, dropout=dropout)
    if fusion_type == "hif":
        return FusionHIF(dim, dropout=dropout)
    raise ValueError(f"Unsupported fusion type '{fusion_type}'. Choose from ['bf','saf','hif'].")
