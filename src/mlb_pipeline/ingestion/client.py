"""Async client for the MLB Stats API."""

import aiohttp
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger()

BASE_URL = "https://statsapi.mlb.com"


class MLBStatsClient:
    """Async HTTP client for statsapi.mlb.com with connection pooling and retry."""

    def __init__(self, session: aiohttp.ClientSession | None = None):
        self._external_session = session is not None
        self._session = session

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                base_url=BASE_URL,
                connector=aiohttp.TCPConnector(limit=20),
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._external_session:
            await self._session.close()

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    async def _get(self, path: str, params: dict | None = None) -> dict:
        session = await self._get_session()
        async with session.get(path, params=params) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def get_schedule(
        self, date: str, sport_id: int = 1, hydrate: str | None = None
    ) -> dict:
        """Get game schedule for a date. Format: YYYY-MM-DD."""
        params: dict = {"sportId": sport_id, "date": date}
        if hydrate:
            params["hydrate"] = hydrate
        return await self._get("/api/v1/schedule", params=params)

    async def get_live_feed(self, game_pk: int) -> dict:
        """Get full live game feed (pitch-by-pitch). This is the primary data source."""
        return await self._get(f"/api/v1.1/game/{game_pk}/feed/live")

    async def get_play_by_play(self, game_pk: int) -> dict:
        """Get play-by-play data for a game."""
        return await self._get(f"/api/v1/game/{game_pk}/playByPlay")

    async def get_boxscore(self, game_pk: int) -> dict:
        """Get boxscore for a game."""
        return await self._get(f"/api/v1/game/{game_pk}/boxscore")

    async def get_linescore(self, game_pk: int) -> dict:
        """Get linescore for a game."""
        return await self._get(f"/api/v1/game/{game_pk}/linescore")

    async def get_teams(self, sport_id: int = 1, season: int | None = None) -> dict:
        """Get all teams."""
        params: dict = {"sportId": sport_id}
        if season:
            params["season"] = season
        return await self._get("/api/v1/teams", params=params)

    async def get_player(self, player_id: int) -> dict:
        """Get player info."""
        return await self._get(f"/api/v1/people/{player_id}")

    async def get_schedule_range(
        self, start: str, end: str, sport_id: int = 1
    ) -> list[dict]:
        """Get all games in a date range. Returns flat list of game entries."""
        data = await self._get(
            "/api/v1/schedule",
            params={"sportId": sport_id, "startDate": start, "endDate": end},
        )
        games = []
        for date_entry in data.get("dates", []):
            games.extend(date_entry.get("games", []))
        return games

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
