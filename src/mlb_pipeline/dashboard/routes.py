"""REST API routes for the MLB dashboard.

Endpoints:
  GET  /api/games                  — today's games with status
  GET  /api/games/{game_pk}        — single game detail
  GET  /api/games/{game_pk}/win_prob — win probability history
  GET  /api/games/{game_pk}/predict  — real-time win prob from state params
  GET  /api/model/status            — loaded model info
  POST /api/model/load              — load a new checkpoint
  GET  /api/health                  — health check
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from mlb_pipeline.dashboard.state import app_state
from mlb_pipeline.ml.inference import WinProbInferenceService, get_inference_service

router = APIRouter()

# Lazy-init inference service
_inference: WinProbInferenceService | None = None


def get_inference() -> WinProbInferenceService:
    global _inference
    if _inference is None:
        _inference = get_inference_service()
    return _inference


# ------------------------------------------------------------------
# Response models
# ------------------------------------------------------------------

class GameSummary(BaseModel):
    game_pk: int
    away_team: str
    home_team: str
    status: str
    away_score: int
    home_score: int
    inning: int | None = None
    home_win_prob: float | None = None


class WinProbPoint(BaseModel):
    inning: int
    half_inning: str
    away_score: int
    home_score: int
    home_win_prob: float
    timestamp: str | None = None


class PredictResponse(BaseModel):
    game_pk: int
    home_win_prob: float
    away_win_prob: float
    model_version: str
    inning: int
    half_inning: str
    score_diff: int


class ModelStatus(BaseModel):
    loaded: bool
    version: str
    device: str


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@router.get("/health")
async def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


@router.get("/games", response_model=list[GameSummary])
async def list_games(query_date: str | None = Query(default=None)):
    """List games for a date (defaults to today)."""
    try:
        from mlb_pipeline.ingestion.client import MLBStatsClient

        d = query_date or str(date.today())
        async with MLBStatsClient() as client:
            schedule = await client.get_schedule(d)

        games = []
        for date_entry in schedule.get("dates", []):
            for g in date_entry.get("games", []):
                game_pk = g["gamePk"]
                linescore = g.get("linescore", {})
                teams = linescore.get("teams", {})
                cached = app_state.win_prob_cache.get(game_pk, [])
                latest_prob = cached[-1]["home_win_prob"] if cached else None

                games.append(GameSummary(
                    game_pk=game_pk,
                    away_team=g["teams"]["away"]["team"]["name"],
                    home_team=g["teams"]["home"]["team"]["name"],
                    status=g["status"]["abstractGameState"],
                    away_score=teams.get("away", {}).get("runs", g["teams"]["away"].get("score", 0)),
                    home_score=teams.get("home", {}).get("runs", g["teams"]["home"].get("score", 0)),
                    inning=linescore.get("currentInning"),
                    home_win_prob=latest_prob,
                ))
        return games
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"MLB API unavailable: {exc}")


@router.get("/games/{game_pk}", response_model=dict)
async def get_game(game_pk: int):
    """Get detailed game state."""
    try:
        from mlb_pipeline.ingestion.client import MLBStatsClient

        async with MLBStatsClient() as client:
            feed = await client.get_live_feed(game_pk)

        game_data = feed.get("gameData", {})
        linescore = feed.get("liveData", {}).get("linescore", {})
        teams_ls = linescore.get("teams", {})

        cached_probs = app_state.win_prob_cache.get(game_pk, [])
        latest_prob = cached_probs[-1]["home_win_prob"] if cached_probs else None

        return {
            "game_pk": game_pk,
            "status": game_data.get("status", {}).get("abstractGameState"),
            "detailed_state": game_data.get("status", {}).get("detailedState"),
            "away_team": game_data.get("teams", {}).get("away", {}).get("name"),
            "home_team": game_data.get("teams", {}).get("home", {}).get("name"),
            "away_score": teams_ls.get("away", {}).get("runs", 0),
            "home_score": teams_ls.get("home", {}).get("runs", 0),
            "inning": linescore.get("currentInning"),
            "half_inning": linescore.get("inningHalf"),
            "outs": linescore.get("outs"),
            "home_win_prob": latest_prob,
            "venue": game_data.get("venue", {}).get("name"),
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.get("/games/{game_pk}/win_prob", response_model=list[WinProbPoint])
async def get_win_prob_history(game_pk: int):
    """Return cached win probability history for a game."""
    cached = app_state.win_prob_cache.get(game_pk, [])
    return [
        WinProbPoint(
            inning=entry.get("inning", 1),
            half_inning=entry.get("half_inning", "top"),
            away_score=entry.get("away_score", 0),
            home_score=entry.get("home_score", 0),
            home_win_prob=entry.get("home_win_prob", 0.5),
        )
        for entry in cached
    ]


@router.get("/games/{game_pk}/predict", response_model=PredictResponse)
async def predict_game(
    game_pk: int,
    inning: int = Query(default=1, ge=1, le=20),
    half_inning: str = Query(default="top"),
    outs: int = Query(default=0, ge=0, le=2),
    away_score: int = Query(default=0, ge=0),
    home_score: int = Query(default=0, ge=0),
    runner_on_first: bool = Query(default=False),
    runner_on_second: bool = Query(default=False),
    runner_on_third: bool = Query(default=False),
):
    """Get win probability prediction for an arbitrary game state."""
    svc = get_inference()
    prob = svc.predict_state(
        inning=inning,
        half_inning=half_inning,
        outs=outs,
        away_score=away_score,
        home_score=home_score,
        runner_on_first=runner_on_first,
        runner_on_second=runner_on_second,
        runner_on_third=runner_on_third,
    )
    return PredictResponse(
        game_pk=game_pk,
        home_win_prob=round(prob, 4),
        away_win_prob=round(1.0 - prob, 4),
        model_version=svc.model_version,
        inning=inning,
        half_inning=half_inning,
        score_diff=home_score - away_score,
    )


@router.get("/model/status", response_model=ModelStatus)
async def model_status():
    svc = get_inference()
    return ModelStatus(
        loaded=svc.is_loaded,
        version=svc.model_version,
        device=str(svc._device),
    )


class LoadModelRequest(BaseModel):
    path: str


@router.post("/model/load")
async def load_model(req: LoadModelRequest):
    global _inference
    _inference = WinProbInferenceService(req.path)
    try:
        _inference.load(req.path)
        return {"status": "loaded", "version": _inference.model_version}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.get("/streams/info")
async def stream_info():
    """Return Redis stream lengths for monitoring."""
    try:
        from mlb_pipeline.stream.producer import EventProducer

        async with EventProducer() as producer:
            lengths = await producer.get_stream_lengths()
        return lengths
    except Exception as exc:
        return {"error": str(exc)}
