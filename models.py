"""PyTorch architectures and loss functions for BA-MAE: Tabular MLP MAE, Huber loss, and MLP Probe."""

from __future__ import annotations
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMSELoss(nn.Module):
    """Mean Squared Error computed strictly over masked feature positions."""

    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Args:

        pred: Reconstructed features of shape (B, D)
        target: Ground truth features of shape (B, D)
        mask: Binary mask of shape (B, D) where 1=masked, 0=visible
        """
        diff_sq = (pred - target) ** 2
        masked_loss = diff_sq * mask
        num_masked = mask.sum()
        return masked_loss.sum() / (num_masked + self.eps)


class MaskedHuberLoss(nn.Module):
    """Smooth L1 / Huber Loss (delta=1.0) computed strictly over masked feature positions."""

    def __init__(self, delta: float = 1.0, eps: float = 1e-8) -> None:
        super().__init__()
        self.delta = delta
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Args:

        pred: Reconstructed features of shape (B, D)
        target: Ground truth features of shape (B, D)
        mask: Binary mask of shape (B, D) where 1=masked, 0=visible
        """
        diff = torch.abs(pred - target)
        huber_elem = torch.where(
            diff < self.delta,
            0.5 * (diff ** 2),
            self.delta * (diff - 0.5 * self.delta),
        )
        masked_loss = huber_elem * mask
        num_masked = mask.sum()
        return masked_loss.sum() / (num_masked + self.eps)


class TabularMLPEncoder(nn.Module):
    """Encoder mapping tabular features (31) to latent representation (128).

    Architecture: 31 -> 256 -> 256 -> 256 -> 128
    LayerNorm, GELU activations, and Dropout 0.10.
    """

    def __init__(self, input_dim: int = 31, latent_dim: int = 128, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def forward_multiscale(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract multi-scale representations:
        Layer 2 output (256-d) and Bottleneck Layer output (128-d).
        net layers:
          0: Linear(input_dim, 256), 1: LayerNorm, 2: GELU, 3: Dropout
          4: Linear(256, 256), 5: LayerNorm, 6: GELU, 7: Dropout (Layer 2)
          8: Linear(256, 256), 9: LayerNorm, 10: GELU, 11: Dropout
          12: Linear(256, 128), 13: LayerNorm, 14: GELU, 15: Dropout (Bottleneck)
        """
        # Block 1
        x1 = self.net[3](self.net[2](self.net[1](self.net[0](x))))
        # Block 2 (Layer 2 representation)
        x2 = self.net[7](self.net[6](self.net[5](self.net[4](x1))))
        # Block 3
        x3 = self.net[11](self.net[10](self.net[9](self.net[8](x2))))
        # Block 4 (Bottleneck representation)
        x4 = self.net[15](self.net[14](self.net[13](self.net[12](x3))))
        return x2, x4


class TabularMLPDecoder(nn.Module):
    """Decoder mapping latent representation (128) back to tabular space (31).

    Architecture: 128 -> 256 -> 31
    LayerNorm, GELU activation, and Dropout 0.10.
    """

    def __init__(self, latent_dim: int = 128, output_dim: int = 31, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(256, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class TabularMAE(nn.Module):
    """Full Tabular Masked Autoencoder combining Encoder and Decoder."""

    def __init__(
        self,
        input_dim: int = 31,
        latent_dim: int = 128,
        dropout: float = 0.10,
        use_mask_token: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.use_mask_token = use_mask_token

        if use_mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, input_dim))
        else:
            self.register_parameter("mask_token", None)

        self.encoder = TabularMLPEncoder(input_dim=input_dim, latent_dim=latent_dim, dropout=dropout)
        self.decoder = TabularMLPDecoder(latent_dim=latent_dim, output_dim=input_dim, dropout=dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args:

        x: Input features (B, input_dim)
        mask: Optional binary mask (B, input_dim), 1=masked, 0=visible
        Returns:
            recon: Reconstructed features (B, input_dim)
            z: Latent embeddings (B, latent_dim)
        """
        if mask is not None:
            if self.mask_token is not None:
                x_in = x * (1.0 - mask) + self.mask_token * mask
            else:
                x_in = x * (1.0 - mask)
        else:
            x_in = x

        z = self.encoder(x_in)
        recon = self.decoder(z)
        return recon, z

    @torch.no_grad()
    def extract_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Extract bottleneck representation z with mask=0 (all features visible)."""
        self.eval()
        return self.encoder(x)

    @torch.no_grad()
    def extract_multiscale_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Extract concatenated multi-scale representation z_multi in R^384.
        Concat Layer 2 (256-d) + Bottleneck Layer (128-d).
        """
        self.eval()
        l2, bottleneck = self.encoder.forward_multiscale(x)
        return torch.cat([l2, bottleneck], dim=-1)


class MLPProbeClassifier(nn.Module):
    """Supervised Linear/MLP Probe trained on frozen representations.

    Architecture: 128 -> 64 -> 1
    Loss: BCEWithLogitsLoss, Decision threshold: p >= 0.50
    """

    def __init__(self, latent_dim: int = 128, hidden_dim: int = 64, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Returns raw logits of shape (B, 1)."""
        return self.net(z)

    def predict_proba(self, z: torch.Tensor) -> torch.Tensor:
        """Returns calibrated probabilities in [0, 1]."""
        logits = self.forward(z)
        return torch.sigmoid(logits)

    def predict(self, z: torch.Tensor, threshold: float = 0.50) -> torch.Tensor:
        """Returns binary predictions (0 or 1)."""
        probs = self.predict_proba(z)
        return (probs >= threshold).long()
