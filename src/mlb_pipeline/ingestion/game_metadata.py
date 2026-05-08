"""Backfill venue_id and first_pitch_time_utc on existing game rows."""

import asyncio
from datetime import datetime
from dataclasses import dataclass

import structlog
from sqlalchemy import select, update
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from mlb_pipeline.ingestion.client import MLBStatsClient
from mlb_pipeline.models.database import Game

logger = structlog.get_logger()


@dataclass
class MetadataStats:
    games_processed: int = 0
    venue_updated: int = 0
    first_pitch_updated: int = 0
    failed: int = 0


def _parse_iso_utc(s: str | None) -> datetime | None:
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)


def _first_pitch_from_feed(feed: dict) -> datetime | None:
    """Use the first play's startTime, falling back to scheduled gameDate."""
    plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
    if plays:
        first_play = plays[0]
        events = first_play.get("playEvents", [])
        if events and events[0].get("startTime"):
            return _parse_iso_utc(events[0]["startTime"])
        if first_play.get("about", {}).get("startTime"):
            return _parse_iso_utc(first_play["about"]["startTime"])

    return _parse_iso_utc(
        feed.get("gameData", {}).get("datetime", {}).get("dateTime")
    )


async def _process_game(
    client: MLBStatsClient,
    engine: Engine,
    game_pk: int,
    sem: asyncio.Semaphore,
    stats: MetadataStats,
) -> None:
    async with sem:
        try:
            feed = await client.get_live_feed(game_pk)
        except Exception as exc:
            stats.failed += 1
            logger.warning("metadata_feed_failed", game_pk=game_pk, error=str(exc))
            return

    venue_id = feed.get("gameData", {}).get("venue", {}).get("id")
    first_pitch = _first_pitch_from_feed(feed)

    updates: dict = {}
    if venue_id is not None:
        updates["venue_id"] = venue_id
    if first_pitch is not None:
        updates["first_pitch_time_utc"] = first_pitch

    if not updates:
        return

    def _save() -> None:
        with Session(engine) as session:
            session.execute(
                update(Game).where(Game.game_pk == game_pk).values(**updates)
            )
            session.commit()

    await asyncio.to_thread(_save)
    stats.games_processed += 1
    if "venue_id" in updates:
        stats.venue_updated += 1
    if "first_pitch_time_utc" in updates:
        stats.first_pitch_updated += 1


async def backfill_game_metadata(
    engine: Engine, concurrency: int = 12, only_missing: bool = True
) -> MetadataStats:
    """For each game, fetch live feed and store venue_id + first_pitch_time_utc."""
    stats = MetadataStats()

    with Session(engine) as session:
        q = select(Game.game_pk).where(Game.status == "Final")
        if only_missing:
            q = q.where(
                (Game.venue_id.is_(None)) | (Game.first_pitch_time_utc.is_(None))
            )
        game_pks = [r[0] for r in session.execute(q).all()]

    logger.info("metadata_backfill_starting", count=len(game_pks))
    if not game_pks:
        return stats

    sem = asyncio.Semaphore(concurrency)

    async with MLBStatsClient() as client:
        chunk_size = 50
        for i in range(0, len(game_pks), chunk_size):
            chunk = game_pks[i : i + chunk_size]
            await asyncio.gather(
                *(_process_game(client, engine, pk, sem, stats) for pk in chunk),
                return_exceptions=False,
            )
            logger.info(
                "metadata_backfill_progress",
                processed=stats.games_processed,
                failed=stats.failed,
                total=len(game_pks),
            )

    logger.info(
        "metadata_backfill_complete",
        processed=stats.games_processed,
        venue_updated=stats.venue_updated,
        first_pitch_updated=stats.first_pitch_updated,
        failed=stats.failed,
    )
    return stats


def venue_ids_referenced_by_games(engine: Engine) -> set[int]:
    """Return the distinct set of non-null venue_ids on games."""
    with Session(engine) as session:
        rows = session.execute(
            select(Game.venue_id).where(Game.venue_id.is_not(None)).distinct()
        ).all()
        return {r[0] for r in rows}
