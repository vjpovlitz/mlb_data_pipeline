"""PyTorch model architectures for win probability prediction.

Two models:
  - WinProbMLP: Multi-layer perceptron with residual connections and dropout.
    Fast, suitable for real-time inference on individual at-bats.

  - WinProbLSTM: Bidirectional LSTM that operates on full game sequences.
    Captures temporal momentum and game flow; better calibrated for late-inning.

Both output a single logit; apply sigmoid for win probability.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class WinProbMLP(nn.Module):
    """Residual MLP for at-bat-level win probability.

    Architecture:
        Input(F) → Linear(256) → [ResBlock × 3] → Linear(128) → Linear(1)

    ResBlock: LayerNorm → Linear → GELU → Dropout → Linear → skip connection
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.res_blocks = nn.ModuleList([
            _ResBlock(hidden_dim, dropout) for _ in range(3)
        ])
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args: x [B, F] → returns logits [B]"""
        h = self.input_proj(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns sigmoid win probability [B]"""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))


class _ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class WinProbLSTM(nn.Module):
    """Bidirectional LSTM for game-sequence win probability.

    Reads a full sequence of at-bat feature vectors [T, F] and predicts
    the win probability after each at-bat.

    Architecture:
        Input(F) → Linear(128) → BiLSTM(2-layer) → Linear(64) → Linear(1)
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.input_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            x: [B, T, F] padded sequence tensor
            lengths: [B] actual lengths for packing (optional)
        Returns:
            logits [B, T] — one per at-bat in the sequence
        """
        B, T, F = x.shape
        proj = self.input_proj(x.view(B * T, F)).view(B, T, -1)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                proj, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.lstm(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        else:
            out, _ = self.lstm(proj)

        logits = self.head(out).squeeze(-1)  # [B, T]
        return logits

    def predict_final(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """Return win probability at the last valid at-bat. Shape: [B]"""
        logits = self.forward(x, lengths)
        if lengths is not None:
            idx = (lengths - 1).clamp(min=0).to(logits.device)
            final_logits = logits[torch.arange(len(idx)), idx]
        else:
            final_logits = logits[:, -1]
        return torch.sigmoid(final_logits)


class EnsembleModel(nn.Module):
    """Weighted ensemble of MLP and LSTM predictions.

    Learns a single scalar weight α ∈ [0,1] (MLP weight); LSTM gets 1-α.
    Can also be used with fixed weights.
    """

    def __init__(self, mlp: WinProbMLP, lstm: WinProbLSTM, learnable_weight: bool = True):
        super().__init__()
        self.mlp = mlp
        self.lstm = lstm
        if learnable_weight:
            self._alpha_logit = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("_alpha_logit", torch.tensor(0.0))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self._alpha_logit)

    def forward(
        self,
        x_flat: torch.Tensor,
        x_seq: torch.Tensor,
        lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            x_flat: [B, F] for MLP
            x_seq:  [B, T, F] for LSTM
            lengths: [B] sequence lengths
        Returns:
            ensemble win probability [B]
        """
        mlp_prob = torch.sigmoid(self.mlp(x_flat))
        lstm_prob = self.lstm.predict_final(x_seq, lengths)
        alpha = self.alpha
        return alpha * mlp_prob + (1 - alpha) * lstm_prob


def build_mlp(feature_dim: int, **kwargs) -> WinProbMLP:
    """Factory with sensible defaults."""
    return WinProbMLP(feature_dim, **kwargs)


def build_lstm(feature_dim: int, **kwargs) -> WinProbLSTM:
    return WinProbLSTM(feature_dim, **kwargs)
