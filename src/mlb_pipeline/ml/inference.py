"""Real-time win probability inference service.

Loads a trained model checkpoint and provides fast synchronous prediction for
the live pipeline. Uses torch.no_grad() and optional CUDA for sub-millisecond
inference latency.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import torch
import structlog

from mlb_pipeline.models.events import AtBatResult
from mlb_pipeline.processing.feature_engineer import extract_features_from_at_bat

logger = structlog.get_logger(__name__)

MODEL_VERSION = "mlp_v1"


class WinProbInferenceService:
    """Singleton-style inference service.

    Load once at startup; call predict() for each at-bat event.
    Thread-safe — predict runs in executor to avoid blocking the event loop.
    """

    def __init__(self, model_path: str | Path | None = None, device: str = "auto"):
        self._model_path = Path(model_path) if model_path else None
        self._device = self._resolve_device(device)
        self._model: torch.nn.Module | None = None
        self._model_version: str = MODEL_VERSION
        self.log = logger.bind(component="inference")

    def load(self, model_path: str | Path | None = None) -> None:
        """Load model from checkpoint. Call before first predict()."""
        path = Path(model_path or self._model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")

        ckpt = torch.load(str(path), map_location=self._device, weights_only=False)
        cfg = ckpt.get("config", {})

        from mlb_pipeline.ml.models import WinProbMLP
        feature_dim = cfg.get("feature_dim", 17)
        hidden_dim = cfg.get("hidden_dim", 256)
        dropout = cfg.get("dropout", 0.2)

        model = WinProbMLP(feature_dim=feature_dim, hidden_dim=hidden_dim, dropout=dropout)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        model.to(self._device)

        self._model = model
        self._model_version = f"mlp_epoch{ckpt.get('epoch', 0)}"
        self.log.info("model_loaded", path=str(path), device=str(self._device), version=self._model_version)

    def load_fallback(self) -> None:
        """Load a simple heuristic fallback (no checkpoint needed).

        Uses a logistic regression proxy based on score difference and inning.
        Used when no trained model is available.
        """
        self._model = _HeuristicModel()
        self._model_version = "heuristic_v1"
        self.log.info("heuristic_model_loaded")

    async def predict(self, event: AtBatResult) -> float:
        """Async wrapper — runs inference in thread executor."""
        return await asyncio.to_thread(self._predict_sync, event)

    def predict_sync(self, event: AtBatResult) -> float:
        """Synchronous prediction for single at-bat event."""
        return self._predict_sync(event)

    def predict_state(
        self,
        inning: int,
        half_inning: str,
        outs: int,
        away_score: int,
        home_score: int,
        runner_on_first: bool = False,
        runner_on_second: bool = False,
        runner_on_third: bool = False,
    ) -> float:
        """Predict from raw game state (useful for the dashboard API)."""
        features = extract_features_from_at_bat(
            inning, half_inning, outs, away_score, home_score,
            runner_on_first, runner_on_second, runner_on_third,
        )
        return self._run_model(features)

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _predict_sync(self, event: AtBatResult) -> float:
        features = extract_features_from_at_bat(
            inning=event.inning,
            half_inning=event.half_inning.value,
            outs=event.outs_after,
            away_score=event.away_score,
            home_score=event.home_score,
            runner_on_first=event.runner_on_first,
            runner_on_second=event.runner_on_second,
            runner_on_third=event.runner_on_third,
        )
        return self._run_model(features)

    def _run_model(self, features) -> float:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() or load_fallback() first.")
        import numpy as np
        x = torch.from_numpy(features.reshape(1, -1)).to(self._device)
        with torch.no_grad():
            if isinstance(self._model, _HeuristicModel):
                return self._model.predict(features)
            prob = torch.sigmoid(self._model(x)).item()
        return float(prob)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)


class _HeuristicModel:
    """Simple heuristic baseline — no ML, just baseball math.

    Based on run expectancy and score-differential models from sabermetrics.
    Provides reasonable default predictions when no trained model exists.
    """

    def predict(self, features) -> float:
        import math
        # features[2] = score_diff (home - away), features[0] = inning_norm
        score_diff = float(features[2])
        inning_norm = float(features[0])
        innings_remaining = float(features[10])

        # Base win prob from score diff
        # Each run is ~15% win probability in the middle of a game
        base = 0.5 + 0.12 * score_diff * (1.0 - innings_remaining * 0.5)
        base = max(0.02, min(0.98, base))
        return base


# Module-level singleton
_inference_service: WinProbInferenceService | None = None


def get_inference_service(model_path: str | Path | None = None) -> WinProbInferenceService:
    """Get or create the module-level inference service singleton."""
    global _inference_service
    if _inference_service is None:
        _inference_service = WinProbInferenceService(model_path)
        if model_path and Path(model_path).exists():
            _inference_service.load(model_path)
        else:
            _inference_service.load_fallback()
    return _inference_service
