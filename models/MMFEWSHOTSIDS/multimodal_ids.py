import torch.nn as nn

from MMFEWSHOTSIDS.g_model import Traffic3DEncoder
from MMFEWSHOTSIDS.s_model import SModel
from MMFEWSHOTSIDS.fusion import build_fusion


class MultiModalIDS(nn.Module):
    def __init__(
        self,
        num_cont: int,
        disc_cardinalities,
        num_classes: int,
        fusion_dim: int = 128,
        fusion_type: str = "hif",
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        g_dropout: float = 0.1,
        s_dropout: float = 0.1,
        fusion_dropout: float = 0.1,
        g_in_dim: int = None,
        A_tensor=None,
    ):
        super().__init__()
 
        self.g = Traffic3DEncoder(
            output_dim=fusion_dim,
            dropout=g_dropout
        )
        self.s = SModel(
            num_cont=num_cont,
            disc_cardinalities=disc_cardinalities,
            output_dim=fusion_dim,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            dropout=s_dropout,
        )
        self.fusion = build_fusion(fusion_type, fusion_dim, nhead=transformer_heads, dropout=fusion_dropout)
        self.cls = nn.Linear(fusion_dim, num_classes)

    def encode(self, g_vec, cont, disc):
        """Return fused embedding before the final classifier."""
        g_feat = self.g(g_vec)
        s_feat = self.s(cont, disc)
        fused = self.fusion(g_feat, s_feat)
        return fused

    def forward(self, g_vec, cont, disc):
        fused = self.encode(g_vec, cont, disc)
        return self.cls(fused)
