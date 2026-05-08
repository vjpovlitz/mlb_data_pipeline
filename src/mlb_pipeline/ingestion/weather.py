"""Weather ingestion via Open-Meteo (free, no auth, historical archive)."""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable

import aiohttp
import structlog
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from mlb_pipeline.models.database import Game, GameWeather, Venue

logger = structlog.get_logger()

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

INDOOR_ROOFS = {"Dome", "Indoor", "Closed"}

HOURLY_FIELDS = [
    "temperature_2m",
    "apparent_temperature",
    "relative_humidity_2m",
    "precipitation",
    "wind_speed_10m",
    "wind_gusts_10m",
    "wind_direction_10m",
    "cloud_cover",
    "pressure_msl",
    "weather_code",
]


@dataclass
class WeatherStats:
    games_total: int = 0
    fetched: int = 0
    indoor_skipped: int = 0
    no_venue: int = 0
    no_first_pitch: int = 0
    failed: int = 0
    failed_game_pks: list[int] = field(default_factory=list)


@dataclass
class _GameWeatherTask:
    game_pk: int
    venue_id: int
    venue_name: str
    latitude: float
    longitude: float
    first_pitch_utc: datetime
    is_indoor: bool


class OpenMeteoClient:
    """Async client for the Open-Meteo historical archive."""

    def __init__(
        self,
        session: aiohttp.ClientSession | None = None,
        concurrency: int = 8,
    ):
        self._external_session = session is not None
        self._session = session
        self._sem = asyncio.Semaphore(concurrency)

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=20),
                timeout=aiohttp.ClientTimeout(total=30),
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._external_session:
            await self._session.close()

    async def fetch_hourly(
        self, latitude: float, longitude: float, date_iso: str
    ) -> dict:
        """Fetch all 24 hours of weather for a single date at a location."""
        async with self._sem:
            session = await self._get_session()
            params = {
                "latitude": latitude,
                "longitude": longitude,
                "start_date": date_iso,
                "end_date": date_iso,
                "hourly": ",".join(HOURLY_FIELDS),
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "timezone": "UTC",
            }
            async with session.get(ARCHIVE_URL, params=params) as resp:
                resp.raise_for_status()
                return await resp.json()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()


def _row_at_hour(hourly: dict, target_iso: str) -> dict | None:
    """Pick the hourly record that matches target_iso (YYYY-MM-DDTHH:00)."""
    times = hourly.get("time", [])
    if target_iso not in times:
        return None
    i = times.index(target_iso)

    def at(key: str):
        vals = hourly.get(key)
        return vals[i] if vals else None

    return {k: at(k) for k in HOURLY_FIELDS}


def _build_weather_row(task: _GameWeatherTask, hourly_row: dict | None) -> GameWeather:
    obs_time = task.first_pitch_utc.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    if task.is_indoor or hourly_row is None:
        return GameWeather(
            game_pk=task.game_pk,
            venue_id=task.venue_id,
            observation_time_utc=obs_time,
            is_indoor=task.is_indoor,
            source="open-meteo",
        )

    return GameWeather(
        game_pk=task.game_pk,
        venue_id=task.venue_id,
        observation_time_utc=obs_time,
        is_indoor=False,
        temperature_f=hourly_row.get("temperature_2m"),
        feels_like_f=hourly_row.get("apparent_temperature"),
        relative_humidity_pct=hourly_row.get("relative_humidity_2m"),
        precipitation_in=hourly_row.get("precipitation"),
        wind_speed_mph=hourly_row.get("wind_speed_10m"),
        wind_gust_mph=hourly_row.get("wind_gusts_10m"),
        wind_direction_deg=hourly_row.get("wind_direction_10m"),
        cloud_cover_pct=hourly_row.get("cloud_cover"),
        pressure_hpa=hourly_row.get("pressure_msl"),
        weather_code=hourly_row.get("weather_code"),
        source="open-meteo",
    )


def _load_game_tasks(
    engine: Engine, game_pks: Iterable[int] | None = None
) -> tuple[list[_GameWeatherTask], WeatherStats]:
    """Build the work list from the games + venues tables."""
    stats = WeatherStats()
    tasks: list[_GameWeatherTask] = []

    with Session(engine) as session:
        q = select(Game, Venue).join(
            Venue, Venue.venue_id == Game.venue_id, isouter=True
        ).where(Game.status == "Final")
        if game_pks is not None:
            q = q.where(Game.game_pk.in_(list(game_pks)))
        rows = session.execute(q).all()

        for game, venue in rows:
            stats.games_total += 1

            if venue is None or venue.latitude is None or venue.longitude is None:
                stats.no_venue += 1
                continue

            if game.first_pitch_time_utc is None:
                stats.no_first_pitch += 1
                continue

            roof = (venue.roof_type or "").strip()
            is_indoor = roof in INDOOR_ROOFS

            tasks.append(
                _GameWeatherTask(
                    game_pk=game.game_pk,
                    venue_id=venue.venue_id,
                    venue_name=venue.name,
                    latitude=venue.latitude,
                    longitude=venue.longitude,
                    first_pitch_utc=(
                        game.first_pitch_time_utc.replace(tzinfo=timezone.utc)
                        if game.first_pitch_time_utc.tzinfo is None
                        else game.first_pitch_time_utc
                    ),
                    is_indoor=is_indoor,
                )
            )

    return tasks, stats


async def _process_task(
    client: OpenMeteoClient, task: _GameWeatherTask, stats: WeatherStats
) -> GameWeather | None:
    target_hour = task.first_pitch_utc.strftime("%Y-%m-%dT%H:00")

    if task.is_indoor:
        stats.indoor_skipped += 1
        return _build_weather_row(task, None)

    try:
        date_iso = task.first_pitch_utc.strftime("%Y-%m-%d")
        payload = await client.fetch_hourly(
            task.latitude, task.longitude, date_iso
        )
    except Exception as exc:
        stats.failed += 1
        stats.failed_game_pks.append(task.game_pk)
        logger.warning("weather_fetch_failed", game_pk=task.game_pk, error=str(exc))
        return None

    hourly_row = _row_at_hour(payload.get("hourly", {}), target_hour)
    stats.fetched += 1
    return _build_weather_row(task, hourly_row)


async def _enrich_one(
    client, engine, game_pk: int, sem: asyncio.Semaphore
) -> bool:
    async with sem:
        try:
            feed = await client.get_live_feed(game_pk)
        except Exception as exc:
            logger.warning("enrich_feed_failed", game_pk=game_pk, error=str(exc))
            return False

    weather = feed.get("gameData", {}).get("weather", {}) or {}
    condition = weather.get("condition")
    temp = weather.get("temp")
    wind = weather.get("wind")
    try:
        mlb_temp = float(temp) if temp not in (None, "") else None
    except (TypeError, ValueError):
        mlb_temp = None

    roof_closed: bool | None = None
    if condition:
        cond_lower = condition.lower()
        if "roof closed" in cond_lower:
            roof_closed = True
        elif "dome" in cond_lower:
            roof_closed = True

    def _save() -> None:
        with Session(engine) as session:
            row = session.get(GameWeather, game_pk)
            if row is None:
                return
            row.mlb_condition = condition
            row.mlb_temp_f = mlb_temp
            row.mlb_wind_text = wind
            if roof_closed is not None:
                row.roof_closed = roof_closed
            elif row.is_indoor:
                row.roof_closed = True
            session.commit()

    await asyncio.to_thread(_save)
    return True


async def enrich_with_mlb_conditions(
    engine: Engine, concurrency: int = 12
) -> int:
    """Pull weather.condition / temp / wind from MLB feed; mark roof_closed for retractables."""
    from mlb_pipeline.ingestion.client import MLBStatsClient

    with Session(engine) as session:
        rows = session.execute(
            select(GameWeather.game_pk).where(GameWeather.mlb_condition.is_(None))
        ).all()
        game_pks = [r[0] for r in rows]

    logger.info("weather_enrich_starting", count=len(game_pks))
    if not game_pks:
        return 0

    sem = asyncio.Semaphore(concurrency)
    enriched = 0
    async with MLBStatsClient() as client:
        chunk_size = 50
        for i in range(0, len(game_pks), chunk_size):
            chunk = game_pks[i : i + chunk_size]
            results = await asyncio.gather(
                *(_enrich_one(client, engine, pk, sem) for pk in chunk),
                return_exceptions=False,
            )
            enriched += sum(1 for r in results if r)
            logger.info(
                "weather_enrich_progress",
                processed=enriched,
                total=len(game_pks),
            )
    logger.info("weather_enrich_complete", enriched=enriched)
    return enriched


async def backfill_weather(
    engine: Engine,
    game_pks: Iterable[int] | None = None,
    concurrency: int = 8,
) -> WeatherStats:
    """Fetch + upsert weather for every Final game (or a subset)."""
    tasks, stats = _load_game_tasks(engine, game_pks)
    logger.info(
        "weather_backfill_starting",
        games_total=stats.games_total,
        fetchable=len(tasks),
        no_venue=stats.no_venue,
        no_first_pitch=stats.no_first_pitch,
    )

    async with OpenMeteoClient(concurrency=concurrency) as client:
        results: list[GameWeather] = []
        chunk_size = 50
        for i in range(0, len(tasks), chunk_size):
            chunk = tasks[i : i + chunk_size]
            chunk_results = await asyncio.gather(
                *(_process_task(client, t, stats) for t in chunk),
                return_exceptions=False,
            )
            results.extend([r for r in chunk_results if r is not None])
            if results:
                with Session(engine) as session:
                    for w in results:
                        session.merge(w)
                    session.commit()
                results.clear()
            logger.info(
                "weather_backfill_progress",
                processed=stats.fetched + stats.indoor_skipped,
                failed=stats.failed,
                total=len(tasks),
            )

    logger.info(
        "weather_backfill_complete",
        games_total=stats.games_total,
        fetched=stats.fetched,
        indoor_skipped=stats.indoor_skipped,
        no_venue=stats.no_venue,
        no_first_pitch=stats.no_first_pitch,
        failed=stats.failed,
    )
    return stats
