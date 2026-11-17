import torch
import torch.nn as nn


class SModel(nn.Module):
    """Transformer-based S-Model with shared & independent embeddings and MLP."""

    def __init__(
        self,
        num_cont: int,
        disc_cardinalities,
        output_dim: int = 128,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_cont <= 0:
            raise ValueError("num_cont must be positive for S-Model.")
        if not disc_cardinalities:
            raise ValueError("disc_cardinalities must be provided for S-Model.")

        self.num_cont = num_cont
        self.num_disc = len(disc_cardinalities)
        self.shared_dim = 112
        self.ind_dim = 16
        self.embed_dim = self.shared_dim + self.ind_dim

        self.output_dim = output_dim
        self.shared_embeddings = nn.ModuleList(
            [nn.Embedding(card, self.shared_dim) for card in disc_cardinalities]
        )
        self.ind_embeddings = nn.ModuleList(
            [nn.Embedding(card, self.ind_dim) for card in disc_cardinalities]
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim,
            nhead=transformer_heads,
            batch_first=True,
            dropout=dropout,
            dim_feedforward=self.embed_dim * 2,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        self.cont_norm = nn.LayerNorm(num_cont)

        mlp_dims = [self.embed_dim * self.num_disc + num_cont, 1108, 2216, output_dim]
        layers = []
        for in_dim, out_dim in zip(mlp_dims[:-1], mlp_dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != output_dim:
                layers.append(nn.ReLU(inplace=True))
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, cont: torch.Tensor, disc: torch.Tensor) -> torch.Tensor:
        embeds = []
        for i in range(self.num_disc):
            shared = self.shared_embeddings[i](disc[:, i])
            unique = self.ind_embeddings[i](disc[:, i])
            embeds.append(torch.cat([shared, unique], dim=-1))
        embed_seq = torch.stack(embeds, dim=1)  # (B, num_disc, embed_dim)
        embed_seq = self.transformer(embed_seq)
        embed_flat = embed_seq.reshape(embed_seq.size(0), -1)

        cont_normed = self.cont_norm(cont)
        combined = torch.cat([embed_flat, cont_normed], dim=-1)
        return self.mlp(combined)
