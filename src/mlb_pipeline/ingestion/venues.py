"""Venue ingestion — fetches lat/lon, roof type, elevation from MLB Stats API."""

import asyncio
from typing import Iterable

import structlog
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from mlb_pipeline.ingestion.client import MLBStatsClient
from mlb_pipeline.models.database import Venue

logger = structlog.get_logger()


async def fetch_venue(client: MLBStatsClient, venue_id: int) -> dict | None:
    """Fetch a single venue with location and field info hydrated."""
    try:
        data = await client._get(
            f"/api/v1/venues/{venue_id}",
            params={"hydrate": "location,fieldInfo"},
        )
    except Exception as exc:
        logger.warning("venue_fetch_failed", venue_id=venue_id, error=str(exc))
        return None

    venues = data.get("venues", [])
    if not venues:
        return None
    return venues[0]


def venue_to_orm(payload: dict) -> Venue:
    location = payload.get("location", {}) or {}
    coords = location.get("defaultCoordinates", {}) or {}
    field = payload.get("fieldInfo", {}) or {}
    return Venue(
        venue_id=payload["id"],
        name=payload.get("name", ""),
        city=location.get("city"),
        state=location.get("stateAbbrev") or location.get("state"),
        country=location.get("country"),
        latitude=coords.get("latitude"),
        longitude=coords.get("longitude"),
        elevation_ft=location.get("elevation"),
        azimuth_angle=location.get("azimuthAngle"),
        roof_type=field.get("roofType"),
        turf_type=field.get("turfType"),
        capacity=field.get("capacity"),
    )


async def ingest_venues(
    client: MLBStatsClient, engine: Engine, venue_ids: Iterable[int]
) -> int:
    """Fetch each venue and upsert. Returns count saved."""
    venue_ids = list({v for v in venue_ids if v})
    logger.info("venue_ingest_starting", count=len(venue_ids))

    payloads = await asyncio.gather(
        *(fetch_venue(client, vid) for vid in venue_ids),
        return_exceptions=False,
    )

    saved = 0
    with Session(engine) as session:
        for payload in payloads:
            if not payload:
                continue
            session.merge(venue_to_orm(payload))
            saved += 1
        session.commit()

    logger.info("venue_ingest_complete", saved=saved)
    return saved
