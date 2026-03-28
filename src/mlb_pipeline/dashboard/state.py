"""Shared application state: WebSocket manager + Redis subscriber."""

from __future__ import annotations

import asyncio
import json
from collections import defaultdict
from pathlib import Path

import structlog
from fastapi import WebSocket

logger = structlog.get_logger(__name__)


class ConnectionManager:
    """Tracks WebSocket connections per game channel."""

    def __init__(self):
        self._connections: dict[str | int, list[WebSocket]] = defaultdict(list)

    async def connect(self, websocket: WebSocket, channel: str | int) -> None:
        await websocket.accept()
        self._connections[channel].append(websocket)
        logger.debug("ws_connected", channel=channel, total=len(self._connections[channel]))

    def disconnect(self, websocket: WebSocket, channel: str | int) -> None:
        conns = self._connections.get(channel, [])
        if websocket in conns:
            conns.remove(websocket)
        logger.debug("ws_disconnected", channel=channel)

    async def broadcast(self, channel: str | int, message: dict) -> None:
        """Send JSON message to all connections on channel and 'all' channel."""
        data = json.dumps(message)
        for ch in [channel, "all"]:
            dead: list[WebSocket] = []
            for ws in list(self._connections.get(ch, [])):
                try:
                    await ws.send_text(data)
                except Exception:
                    dead.append(ws)
            for ws in dead:
                self.disconnect(ws, ch)

    def connection_count(self, channel: str | int | None = None) -> int:
        if channel is not None:
            return len(self._connections.get(channel, []))
        return sum(len(v) for v in self._connections.values())


class AppState:
    """Central application state container."""

    def __init__(self):
        self.ws_manager = ConnectionManager()
        self._redis_task: asyncio.Task | None = None
        self._redis = None
        self.active_games: dict[int, dict] = {}
        self.win_prob_cache: dict[int, list[dict]] = defaultdict(list)
        self.log = logger.bind(component="app_state")

    async def start(self) -> None:
        """Start background Redis subscriber task."""
        try:
            import redis.asyncio as aioredis
            from mlb_pipeline.config import settings

            self._redis = await aioredis.from_url(settings.redis_url, decode_responses=True)
            self._redis_task = asyncio.create_task(self._redis_subscriber(), name="redis_sub")
            self.log.info("app_state_started")
        except Exception as exc:
            self.log.warning("redis_unavailable", error=str(exc))

    async def stop(self) -> None:
        if self._redis_task:
            self._redis_task.cancel()
            try:
                await self._redis_task
            except asyncio.CancelledError:
                pass
        if self._redis:
            await self._redis.aclose()

    async def _redis_subscriber(self) -> None:
        """Poll Redis streams and broadcast to WebSocket clients."""
        from mlb_pipeline.stream.producer import STREAM_AT_BATS, STREAM_PITCHES

        stream_positions = {
            STREAM_PITCHES: "$",
            STREAM_AT_BATS: "$",
            "mlb:win_prob": "$",
            "mlb:game_state": "$",
        }

        self.log.info("redis_subscriber_started")
        while True:
            try:
                results = await self._redis.xread(stream_positions, count=20, block=500)
                if not results:
                    continue
                for stream_name, messages in results:
                    for msg_id, fields in messages:
                        stream_positions[stream_name] = msg_id
                        await self._handle_stream_message(stream_name, fields.get("data", ""))
            except asyncio.CancelledError:
                break
            except Exception as exc:
                self.log.error("redis_subscriber_error", error=str(exc))
                await asyncio.sleep(1)

    async def _handle_stream_message(self, stream: str, data: str) -> None:
        if not data:
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            return

        game_pk = payload.get("game_pk")
        if not game_pk:
            return

        if stream == "mlb:win_prob":
            entry = {
                "type": "win_prob",
                "game_pk": game_pk,
                "home_win_prob": payload.get("home_win_prob"),
                "inning": payload.get("inning"),
                "half_inning": payload.get("half_inning"),
                "away_score": payload.get("away_score"),
                "home_score": payload.get("home_score"),
            }
            self.win_prob_cache[game_pk].append(entry)
            # Keep last 100 per game
            if len(self.win_prob_cache[game_pk]) > 100:
                self.win_prob_cache[game_pk] = self.win_prob_cache[game_pk][-100:]
            await self.ws_manager.broadcast(game_pk, entry)

        elif stream == "mlb:at_bats":
            payload["type"] = "at_bat"
            await self.ws_manager.broadcast(game_pk, payload)

        elif "pitches" in stream:
            payload["type"] = "pitch"
            await self.ws_manager.broadcast(game_pk, payload)

        elif "game_state" in stream:
            payload["type"] = "game_state"
            if game_pk not in self.active_games:
                self.active_games[game_pk] = {}
            self.active_games[game_pk].update(payload)
            await self.ws_manager.broadcast(game_pk, payload)


app_state = AppState()
