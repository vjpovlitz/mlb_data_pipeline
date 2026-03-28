"""FastAPI application for the MLB win probability dashboard.

Provides:
  - REST API for games, win probability history, model status
  - WebSocket endpoint for real-time pitch/at-bat/probability pushes
  - Static file serving for the frontend HTML/JS
  - Redis pub-sub bridge to push events to connected browsers
"""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from mlb_pipeline.dashboard.routes import router
from mlb_pipeline.dashboard.state import AppState, app_state

logger = structlog.get_logger(__name__)

FRONTEND_DIR = Path(__file__).parent.parent.parent.parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start/stop background tasks."""
    await app_state.start()
    yield
    await app_state.stop()


def create_app() -> FastAPI:
    application = FastAPI(
        title="MLB Win Probability Dashboard",
        description="Real-time MLB game win probability predictions powered by PyTorch.",
        version="2.0.0",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(router, prefix="/api")

    # WebSocket endpoint for real-time updates
    @application.websocket("/ws/{game_pk}")
    async def game_websocket(websocket: WebSocket, game_pk: int):
        await app_state.ws_manager.connect(websocket, game_pk)
        try:
            while True:
                # Keep connection alive; server pushes data
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass
        finally:
            app_state.ws_manager.disconnect(websocket, game_pk)

    @application.websocket("/ws/all")
    async def all_games_websocket(websocket: WebSocket):
        await app_state.ws_manager.connect(websocket, channel="all")
        try:
            while True:
                data = await asyncio.wait_for(websocket.receive_text(), timeout=30.0)
                if data == "ping":
                    await websocket.send_text("pong")
        except (WebSocketDisconnect, asyncio.TimeoutError):
            pass
        finally:
            app_state.ws_manager.disconnect(websocket, "all")

    # Serve frontend
    if FRONTEND_DIR.exists():
        application.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

        @application.get("/")
        async def serve_frontend():
            return FileResponse(str(FRONTEND_DIR / "index.html"))

    return application


app = create_app()
