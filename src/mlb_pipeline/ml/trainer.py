"""Training loop for win probability models.

Supports:
  - MLP training on flat at-bat rows
  - LSTM training on per-game sequences
  - GPU via CUDA if available
  - Checkpointing, early stopping, and evaluation metrics
  - Learning rate scheduling (ReduceLROnPlateau)
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

import structlog

from mlb_pipeline.config import settings
from mlb_pipeline.ml.dataset import WinProbDataset, WinProbSequenceDataset
from mlb_pipeline.ml.models import WinProbLSTM, WinProbMLP

logger = structlog.get_logger(__name__)


@dataclass
class TrainingConfig:
    model_type: str = "mlp"          # "mlp" | "lstm"
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    dropout: float = 0.2
    hidden_dim: int = 256
    lstm_hidden_dim: int = 128
    lstm_layers: int = 2
    patience: int = 10               # early stopping
    min_delta: float = 1e-4
    grad_clip: float = 1.0
    train_frac: float = 0.8
    num_workers: int = 2
    seed: int = 42
    device: str = "auto"             # "auto" | "cuda" | "cpu"


@dataclass
class TrainingMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_auc: float
    val_brier: float
    val_logloss: float
    lr: float
    elapsed_s: float


@dataclass
class TrainingResult:
    model_path: str
    metrics_path: str
    best_val_auc: float
    best_epoch: int
    total_epochs: int
    history: list[dict] = field(default_factory=list)


class ModelTrainer:
    """Manages the full training lifecycle for MLP or LSTM models."""

    def __init__(self, config: TrainingConfig | None = None):
        self.cfg = config or TrainingConfig()
        self.device = self._resolve_device()
        self.log = logger.bind(component="trainer", model_type=self.cfg.model_type)

    def train_mlp(self, dataset_path: str, output_dir: str | None = None) -> TrainingResult:
        """Train the MLP model from a Parquet training dataset."""
        torch.manual_seed(self.cfg.seed)
        df = _load_parquet(dataset_path)
        train_ds, val_ds = WinProbDataset.temporal_split(df, self.cfg.train_frac)
        self.log.info("dataset_loaded", train=len(train_ds), val=len(val_ds), features=train_ds.feature_dim)

        model = WinProbMLP(
            feature_dim=train_ds.feature_dim,
            hidden_dim=self.cfg.hidden_dim,
            dropout=self.cfg.dropout,
        ).to(self.device)

        return self._run_training(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            output_dir=output_dir or str(settings.model_dir),
            name=f"mlp_{int(time.time())}",
        )

    def train_lstm(self, dataset_path: str, output_dir: str | None = None) -> TrainingResult:
        """Train the LSTM model from a Parquet training dataset."""
        torch.manual_seed(self.cfg.seed)
        df = _load_parquet(dataset_path)
        train_ds, val_ds = WinProbSequenceDataset.temporal_split(df, self.cfg.train_frac)
        self.log.info("sequence_dataset_loaded", train=len(train_ds), val=len(val_ds))

        model = WinProbLSTM(
            feature_dim=train_ds.feature_dim,
            hidden_dim=self.cfg.lstm_hidden_dim,
            num_layers=self.cfg.lstm_layers,
            dropout=self.cfg.dropout,
        ).to(self.device)

        return self._run_training(
            model=model,
            train_ds=train_ds,
            val_ds=val_ds,
            output_dir=output_dir or str(settings.model_dir),
            name=f"lstm_{int(time.time())}",
            is_sequence=True,
        )

    # ------------------------------------------------------------------
    # Core training loop
    # ------------------------------------------------------------------

    def _run_training(
        self,
        model: nn.Module,
        train_ds,
        val_ds,
        output_dir: str,
        name: str,
        is_sequence: bool = False,
    ) -> TrainingResult:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        optimizer = AdamW(model.parameters(), lr=self.cfg.learning_rate, weight_decay=self.cfg.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, verbose=False)
        criterion = nn.BCEWithLogitsLoss()

        train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=self.cfg.batch_size * 2,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

        best_auc = 0.0
        best_epoch = 0
        no_improve = 0
        history: list[dict] = []

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_loss = self._train_epoch(model, train_loader, optimizer, criterion, is_sequence)
            val_loss, val_preds, val_labels = self._eval_epoch(model, val_loader, criterion, is_sequence)

            metrics = self._compute_metrics(
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                preds=val_preds,
                labels=val_labels,
                lr=optimizer.param_groups[0]["lr"],
                elapsed=time.time() - t0,
            )
            history.append(asdict(metrics))
            scheduler.step(metrics.val_auc)

            self.log.info(
                "epoch_done",
                epoch=epoch,
                train_loss=f"{metrics.train_loss:.4f}",
                val_loss=f"{metrics.val_loss:.4f}",
                val_auc=f"{metrics.val_auc:.4f}",
                val_brier=f"{metrics.val_brier:.4f}",
            )

            if metrics.val_auc > best_auc + self.cfg.min_delta:
                best_auc = metrics.val_auc
                best_epoch = epoch
                no_improve = 0
                model_path = out_path / f"{name}_best.pt"
                torch.save({
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "config": asdict(self.cfg),
                    "metrics": asdict(metrics),
                }, model_path)
            else:
                no_improve += 1
                if no_improve >= self.cfg.patience:
                    self.log.info("early_stopping", epoch=epoch, best_epoch=best_epoch)
                    break

        metrics_path = out_path / f"{name}_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(history, f, indent=2)

        self.log.info("training_complete", best_auc=f"{best_auc:.4f}", best_epoch=best_epoch)
        return TrainingResult(
            model_path=str(model_path),
            metrics_path=str(metrics_path),
            best_val_auc=best_auc,
            best_epoch=best_epoch,
            total_epochs=epoch,
            history=history,
        )

    def _train_epoch(self, model, loader, optimizer, criterion, is_sequence: bool) -> float:
        model.train()
        total_loss = 0.0
        for batch in loader:
            optimizer.zero_grad()
            if is_sequence:
                x, y, lengths = batch
                x = x.to(self.device)
                y = y.to(self.device)
                lengths = lengths.to(self.device)
                # Use final-step logit for sequence training
                logits_all = model(x, lengths)
                idx = (lengths - 1).clamp(min=0)
                logits = logits_all[torch.arange(len(idx)), idx]
            else:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                logits = model(x)

            loss = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
            optimizer.step()
            total_loss += loss.item() * len(y)

        return total_loss / len(loader.dataset)

    @torch.no_grad()
    def _eval_epoch(self, model, loader, criterion, is_sequence: bool):
        model.eval()
        total_loss = 0.0
        all_preds: list[float] = []
        all_labels: list[float] = []

        for batch in loader:
            if is_sequence:
                x, y, lengths = batch
                x = x.to(self.device)
                y = y.to(self.device)
                lengths = lengths.to(self.device)
                logits_all = model(x, lengths)
                idx = (lengths - 1).clamp(min=0)
                logits = logits_all[torch.arange(len(idx)), idx]
            else:
                x, y = batch
                x = x.to(self.device)
                y = y.to(self.device)
                logits = model(x)

            loss = criterion(logits, y)
            total_loss += loss.item() * len(y)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.extend(probs.tolist())
            all_labels.extend(y.cpu().numpy().tolist())

        return total_loss / len(loader.dataset), np.array(all_preds), np.array(all_labels)

    def _compute_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        preds: np.ndarray,
        labels: np.ndarray,
        lr: float,
        elapsed: float,
    ) -> TrainingMetrics:
        try:
            auc = roc_auc_score(labels, preds)
        except Exception:
            auc = 0.5
        brier = brier_score_loss(labels, preds)
        ll = log_loss(labels, preds)
        return TrainingMetrics(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            val_auc=auc,
            val_brier=brier,
            val_logloss=ll,
            lr=lr,
            elapsed_s=elapsed,
        )

    def _resolve_device(self) -> torch.device:
        if self.cfg.device == "auto":
            if torch.cuda.is_available():
                d = torch.device("cuda")
                self.log.info("using_gpu", name=torch.cuda.get_device_name(0))
            else:
                d = torch.device("cpu")
                self.log.info("using_cpu")
        else:
            d = torch.device(self.cfg.device)
        return d


# ------------------------------------------------------------------
# Model evaluator
# ------------------------------------------------------------------

class ModelEvaluator:
    """Load a checkpoint and evaluate on held-out data."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.saved_config = ckpt.get("config", {})
        self.saved_metrics = ckpt.get("metrics", {})
        self.log = logger.bind(component="evaluator")

    def evaluate(self, dataset_path: str, model_type: str = "mlp") -> dict:
        """Run full evaluation including calibration analysis."""
        import polars as pl

        df = pl.read_parquet(dataset_path)
        game_pks = sorted(df["game_pk"].unique().to_list())
        split_idx = int(len(game_pks) * 0.8)
        test_pks = set(game_pks[split_idx:])
        test_df = df.filter(pl.col("game_pk").is_in(test_pks))

        if model_type == "mlp":
            ds = WinProbDataset(test_df)
            model = WinProbMLP(feature_dim=ds.feature_dim).to("cpu")
        else:
            ds = WinProbSequenceDataset(test_df)
            model = WinProbLSTM(feature_dim=ds.feature_dim).to("cpu")

        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        model.eval()

        loader = DataLoader(ds, batch_size=512, shuffle=False)
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                if model_type == "lstm":
                    x, y, lengths = batch
                    logits_all = model(x, lengths)
                    idx = (lengths - 1).clamp(min=0)
                    logits = logits_all[torch.arange(len(idx)), idx]
                else:
                    x, y = batch
                    logits = model(x)
                probs = torch.sigmoid(logits).numpy()
                all_preds.extend(probs.tolist())
                all_labels.extend(y.numpy().tolist())

        preds = np.array(all_preds)
        labels = np.array(all_labels)

        fraction_of_pos, mean_pred = calibration_curve(labels, preds, n_bins=10)

        results = {
            "auc": float(roc_auc_score(labels, preds)),
            "brier_score": float(brier_score_loss(labels, preds)),
            "log_loss": float(log_loss(labels, preds)),
            "n_test_samples": len(labels),
            "n_test_games": len(test_pks),
            "calibration": {
                "fraction_of_positives": fraction_of_pos.tolist(),
                "mean_predicted_value": mean_pred.tolist(),
            },
        }
        self.log.info("evaluation_complete", **{k: v for k, v in results.items() if k != "calibration"})
        return results


def _load_parquet(path: str):
    import polars as pl
    return pl.read_parquet(path)
