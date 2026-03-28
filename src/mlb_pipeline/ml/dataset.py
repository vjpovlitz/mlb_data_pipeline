"""PyTorch datasets for win probability training.

Two dataset types:
  - WinProbDataset: flat MLP dataset — one row per at-bat
  - WinProbSequenceDataset: LSTM dataset — one sequence per game (at-bats ordered by time)
"""

from __future__ import annotations

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from mlb_pipeline.processing.feature_engineer import TRAINING_FEATURE_COLS


class WinProbDataset(Dataset):
    """Flat at-bat dataset for MLP training.

    Each sample is (feature_vector, label) where label=1 means home team won.
    """

    def __init__(self, df: pl.DataFrame):
        """
        Args:
            df: Output of build_training_dataset(), contains feature cols + label.
        """
        features = df.select(TRAINING_FEATURE_COLS).to_numpy(allow_copy=True).astype(np.float32)
        labels = df.select("label").to_numpy(allow_copy=True).astype(np.float32).squeeze()

        self.X = torch.from_numpy(features)
        self.y = torch.from_numpy(labels)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

    @property
    def feature_dim(self) -> int:
        return self.X.shape[1]

    @classmethod
    def from_parquet(cls, path: str) -> "WinProbDataset":
        df = pl.read_parquet(path)
        return cls(df)

    @classmethod
    def temporal_split(
        cls,
        df: pl.DataFrame,
        train_frac: float = 0.8,
    ) -> tuple["WinProbDataset", "WinProbDataset"]:
        """Split by game_pk sorted order — avoids data leakage across games."""
        game_pks = sorted(df["game_pk"].unique().to_list())
        split_idx = int(len(game_pks) * train_frac)
        train_pks = set(game_pks[:split_idx])
        test_pks = set(game_pks[split_idx:])

        train_df = df.filter(pl.col("game_pk").is_in(train_pks))
        test_df = df.filter(pl.col("game_pk").is_in(test_pks))
        return cls(train_df), cls(test_df)


class WinProbSequenceDataset(Dataset):
    """Per-game sequence dataset for LSTM training.

    Each sample is (sequence_tensor [T, F], label) where T is the number of
    at-bats in the game and F is feature_dim. Sequences are padded to max_len.
    """

    def __init__(self, df: pl.DataFrame, max_len: int = 80):
        """
        Args:
            df: Output of build_training_dataset(), sorted by game_pk + at_bat_index.
            max_len: Pad/truncate all sequences to this length.
        """
        self.max_len = max_len
        self.sequences: list[torch.Tensor] = []
        self.lengths: list[int] = []
        self.labels: list[torch.Tensor] = []

        grouped = df.sort(["game_pk", "at_bat_index"]).group_by("game_pk")
        for _, game_df in grouped:
            game_df = game_df.sort("at_bat_index")
            feats = game_df.select(TRAINING_FEATURE_COLS).to_numpy(allow_copy=True).astype(np.float32)
            label = float(game_df["label"].mean())  # same for all rows in game

            T = min(len(feats), max_len)
            padded = np.zeros((max_len, feats.shape[1]), dtype=np.float32)
            padded[:T] = feats[:T]

            self.sequences.append(torch.from_numpy(padded))
            self.lengths.append(T)
            self.labels.append(torch.tensor(label, dtype=torch.float32))

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        return self.sequences[idx], self.labels[idx], self.lengths[idx]

    @property
    def feature_dim(self) -> int:
        return self.sequences[0].shape[1] if self.sequences else 0

    @classmethod
    def temporal_split(
        cls,
        df: pl.DataFrame,
        train_frac: float = 0.8,
        max_len: int = 80,
    ) -> tuple["WinProbSequenceDataset", "WinProbSequenceDataset"]:
        game_pks = sorted(df["game_pk"].unique().to_list())
        split_idx = int(len(game_pks) * train_frac)
        train_pks = set(game_pks[:split_idx])
        test_pks = set(game_pks[split_idx:])

        return (
            cls(df.filter(pl.col("game_pk").is_in(train_pks)), max_len=max_len),
            cls(df.filter(pl.col("game_pk").is_in(test_pks)), max_len=max_len),
        )
