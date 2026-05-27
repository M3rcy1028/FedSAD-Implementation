from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class ViewBranch(nn.Module):
    """Lightweight 1D CNN branch for a single view."""

    def __init__(self, in_len: int, out_channels: int, dropout: float):
        super().__init__()
        self.conv1 = nn.Conv1d(1, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(out_channels * in_len, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.bn1(self.conv1(x)))
        z = z.flatten(1)
        z = self.dropout(z)
        z = self.fc(z)
        return z


class MultiViewKnowledgeNet(nn.Module):
    """Two-level fusion: view-wise soft attention followed by KG-informed fusion."""

    def __init__(
        self,
        view_lengths: Sequence[int],
        cnn_channels: int,
        fusion_hidden: int,
        kg_dim: int = 0,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.branches = nn.ModuleList(
            [ViewBranch(in_len, cnn_channels, dropout) for in_len in view_lengths]
        )
        self.attn_projections = nn.ModuleList(
            [nn.Linear(cnn_channels, 1) for _ in view_lengths]
        )
        self.softmax = nn.Softmax(dim=1)
        self.post_view_dropout = nn.Dropout(p=dropout)
        self.use_kg = kg_dim > 0
        if self.use_kg:
            kg_out_dim = cnn_channels
            self.kg_fc = nn.Sequential(
                nn.Linear(kg_dim, kg_out_dim),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            )
            lstm_input_dim=cnn_channels * 2
        else:
            kg_out_dim = 0
            self.kg_fc = None
            lstm_input_dim=cnn_channels


        self.lstm = nn.LSTM(
            input_size = lstm_input_dim,   # input feature size
            hidden_size=cnn_channels,  # output feature size
            batch_first=True
        )

        self.temporal_attn = nn.Linear(cnn_channels, 1)

        fusion_in = cnn_channels
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(fusion_in, fusion_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
        )
        self.classifier = nn.Linear(fusion_hidden, 1)

    # def forward(self, views: Sequence[torch.Tensor], kg_vec: torch.Tensor | None = None) -> torch.Tensor:
    #     view_feats = [branch(x) for branch, x in zip(self.branches, views)]  # each [B, C]
    #     stacked_feats = torch.stack(view_feats, dim=1)  # [B, V, C]

    #     scores = torch.stack(
    #         [proj(feat) for proj, feat in zip(self.attn_projections, view_feats)],
    #         dim=1,
    #     )  # [B, V, 1]
    #     alpha = self.softmax(scores)
    #     fused_views = (alpha * stacked_feats).sum(dim=1)  # [B, C]
    #     fused_views = self.post_view_dropout(fused_views)

    #     fusion_input = fused_views
    #     if self.use_kg:
    #         if kg_vec is None:
    #             raise ValueError("KG vector required but not provided.")
    #         if kg_vec.ndim == 1:
    #             kg_vec = kg_vec.unsqueeze(0)
    #         kg_embed = self.kg_fc(kg_vec)
    #         fusion_input = torch.cat([fused_views, kg_embed], dim=-1)

    #     fusion = self.fusion_fc(fusion_input)
    #     logits = self.classifier(fusion)
    #     return logits.squeeze(1)

    def forward(self, views, kg_vec=None):
        view_feats = [branch(x) for branch, x in zip(self.branches, views)]  # [B, C]
        stacked_feats = torch.stack(view_feats, dim=1)  # [B, V, C]

        if self.use_kg:
            if kg_vec is None:
                raise ValueError("KG vector required but not provided.")
            if kg_vec.ndim == 1:
                kg_vec = kg_vec.unsqueeze(0)

            kg_embed = self.kg_fc(kg_vec)  # [B, C]

            # expand KG to match sequence
            kg_expand = kg_embed.unsqueeze(1).expand(-1, stacked_feats.size(1), -1)  # [B, V, C]

            # paper Eq.(4): concat
            fused_input = torch.cat([stacked_feats, kg_expand], dim=-1)  # [B, V, 2C]
        else:
            fused_input = stacked_feats

        # LSTM (after fusion)
        lstm_out, _ = self.lstm(fused_input)  # [B, V, H]

        # Attention
        attn_scores = self.temporal_attn(lstm_out)  # [B, V, 1]
        attn_weights = torch.softmax(attn_scores, dim=1)
        fused = (attn_weights * lstm_out).sum(dim=1)  # [B, H]

        fused = self.post_view_dropout(fused)

        fusion = self.fusion_fc(fused)
        logits = self.classifier(fusion)
        return logits.squeeze(1)