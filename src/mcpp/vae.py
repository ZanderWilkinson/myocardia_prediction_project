"""Compact supervised spatiotemporal VAE; demographics enter only the MI head."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class VAEConfig:
    n_frames: int = 10
    n_points: int = 384
    n_regions: int = 3
    latent_dim: int = 15
    width: int = 384
    heads: int = 8
    spatial_tokens: int = 16
    feedforward: int = 1024
    encoder_depth: int = 2
    decoder_depth: int = 1
    demographic_dim: int = 7
    dropout: float = .25


class AxialBlock(nn.Module):
    """Spatial attention per frame followed by temporal attention per spatial slot."""

    def __init__(self, config):
        super().__init__()
        def layer():
            return nn.TransformerEncoderLayer(
                config.width, config.heads, config.feedforward, config.dropout,
                activation="gelu", batch_first=True, norm_first=True,
            )
        self.spatial = layer()
        self.temporal = layer()

    def forward(self, x):
        b, t, s, d = x.shape
        x = self.spatial(x.reshape(b * t, s, d)).reshape(b, t, s, d)
        x = x.transpose(1, 2).reshape(b * s, t, d)
        x = self.temporal(x).reshape(b, s, t, d)
        return x.transpose(1, 2)


class CardiacVAE(nn.Module):
    """XYZ + anatomical IDs -> Gaussian latent; [mean, demographics] -> MI logit.

    Learned spatial queries attend to all sampled points in each frame. Axial
    blocks gather space/time interactions with short sequences. The decoder
    reconstructs ordered XYZ from z alone, with no encoder shortcuts. Region
    IDs are categorical context, not a regression target or the MI diagnosis.
    """

    def __init__(self, config=VAEConfig()):
        super().__init__()
        self.config = config
        if (config.n_points < 1 or config.spatial_tokens < 1
                or config.n_points % config.spatial_tokens
                or config.width % config.heads or config.latent_dim < 1
                or config.encoder_depth < 1 or config.decoder_depth < 1):
            raise ValueError("Use positive dimensions, width divisible by heads and points by tokens.")
        d, s, t = config.width, config.spatial_tokens, config.n_frames
        self.region_embedding = nn.Embedding(config.n_regions, 16)
        self.point_projection = nn.Sequential(nn.Linear(19, d), nn.GELU(), nn.LayerNorm(d))
        self.point_position = nn.Parameter(torch.randn(1, 1, config.n_points, d) * .02)
        self.spatial_queries = nn.Parameter(torch.randn(1, s, d) * .02)
        self.pool_attention = nn.MultiheadAttention(d, config.heads, config.dropout, batch_first=True)
        self.pool_norm = nn.LayerNorm(d)
        self.time_position = nn.Parameter(torch.randn(1, t, 1, d) * .02)
        self.encoder = nn.Sequential(*[AxialBlock(config) for _ in range(config.encoder_depth)])
        self.encoder_norm = nn.LayerNorm(d)
        self.mu = nn.Linear(d, config.latent_dim)
        self.logvar = nn.Linear(d, config.latent_dim)
        self.latent_projection = nn.Linear(config.latent_dim, t * s * d)
        self.decoder = nn.Sequential(*[AxialBlock(config) for _ in range(config.decoder_depth)])
        self.decoder_norm = nn.LayerNorm(d)
        self.xyz_projection = nn.Linear(d, config.n_points // s * 3)
        self.classifier = nn.Sequential(
            nn.Linear(config.latent_dim + config.demographic_dim, 128), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(128, 64), nn.GELU(),
            nn.Dropout(config.dropout), nn.Linear(64, 1),
        )

    def encode(self, xyz, regions):
        b, t, n, _ = xyz.shape
        if (t, n, xyz.shape[-1]) != (self.config.n_frames, self.config.n_points, 3):
            raise ValueError("XYZ shape must match (batch, configured frames, configured points, 3).")
        if regions.shape != (b, n):
            raise ValueError("Region IDs must have shape (batch, points).")
        embeddings = self.region_embedding(regions)[:, None].expand(-1, t, -1, -1)
        points = self.point_projection(torch.cat((xyz, embeddings), dim=-1)) + self.point_position
        points = points.reshape(b * t, n, self.config.width)
        queries = self.spatial_queries.expand(b * t, -1, -1)
        pooled, _ = self.pool_attention(queries, points, points, need_weights=False)
        x = self.pool_norm(pooled + queries).reshape(b, t, self.config.spatial_tokens, -1)
        x = self.encoder(x + self.time_position)
        summary = self.encoder_norm(x).mean(dim=(1, 2))
        return self.mu(summary), self.logvar(summary).clamp(-12, 8)

    def decode(self, z):
        c = self.config
        x = self.latent_projection(z).reshape(-1, c.n_frames, c.spatial_tokens, c.width)
        x = self.decoder(x + self.time_position)
        return self.xyz_projection(self.decoder_norm(x)).reshape(-1, c.n_frames, c.n_points, 3)

    def classify(self, mu, demographics):
        if demographics.shape != (mu.shape[0], self.config.demographic_dim):
            raise ValueError("Demographic shape must match batch and configured demographic_dim.")
        return self.classifier(torch.cat((mu, demographics), dim=-1)).squeeze(-1)

    def forward(self, xyz, regions, demographics, *, sample=None):
        mu, logvar = self.encode(xyz, regions)
        sample = self.training if sample is None else sample
        z = mu + torch.randn_like(mu) * torch.exp(.5 * logvar) if sample else mu
        return {"reconstruction": self.decode(z), "mu": mu, "logvar": logvar,
                "logits": self.classify(mu, demographics)}

    def parameter_count(self):
        return sum(parameter.numel() for parameter in self.parameters())


def vae_loss(output, xyz, labels, *, beta=.01, classification_weight=1., pos_weight=None):
    """Mean XYZ MSE + beta * mean per-dimension KL + weighted binary CE.

    KL is averaged over patients AND latent dimensions; beta therefore does not
    silently change strength with point count. Warm beta during early training.
    The head uses mu for stable classification while reconstruction samples z.
    """
    # FP16 squaring/exp can overflow well before the true loss becomes large.
    # Keep these reductions in FP32 even when the model uses CUDA autocast.
    reconstruction = F.mse_loss(output["reconstruction"].float(), xyz.float())
    mu, logvar = output["mu"].float(), output["logvar"].float()
    kl = -.5 * (1 + logvar - mu.square() - logvar.exp()).mean()
    classification = F.binary_cross_entropy_with_logits(
        output["logits"].float(), labels.float(), pos_weight=pos_weight,
    )
    return {"loss": reconstruction + beta * kl + classification_weight * classification,
            "reconstruction": reconstruction, "kl": kl, "classification": classification}
