"""Verify SQL Server has every Final game from the MLB schedule for a date range.

Usage: python scripts/verify_coverage.py 2026-03-26 2026-05-07
"""

import asyncio
import sys
from collections import defaultdict
from datetime import date

from sqlalchemy import create_engine, func, select

from mlb_pipeline.config import settings
from mlb_pipeline.ingestion.client import MLBStatsClient
from mlb_pipeline.models.database import Game, Pitch
from sqlalchemy.orm import Session


async def main(start: date, end: date) -> int:
    async with MLBStatsClient() as client:
        sched = await client.get_schedule_range(start.isoformat(), end.isoformat())

    by_date_api: dict[str, set[int]] = defaultdict(set)
    final_pks: set[int] = set()
    nonfinal_by_state: dict[str, list[int]] = defaultdict(list)
    skipped_detailed: dict[str, list[int]] = defaultdict(list)
    for g in sched:
        d = g["gameDate"][:10]
        pk = g["gamePk"]
        state = g["status"]["abstractGameState"]
        detailed = g["status"].get("detailedState", "")
        by_date_api[d].add(pk)
        if state == "Final" and detailed not in {"Postponed", "Cancelled", "Suspended"}:
            final_pks.add(pk)
        elif detailed in {"Postponed", "Cancelled", "Suspended"}:
            skipped_detailed[detailed].append(pk)
        else:
            nonfinal_by_state[state].append(pk)

    engine = create_engine(settings.db_connection_string)
    with Session(engine) as session:
        rows = session.execute(
            select(Game.game_pk, Game.game_date, Game.status).where(
                Game.game_date >= start, Game.game_date <= end
            )
        ).all()
        db_pks = {r[0] for r in rows}
        db_status_by_pk = {r[0]: r[2] for r in rows}

        pitch_counts = dict(
            session.execute(
                select(Pitch.game_pk, func.count(Pitch.event_id))
                .where(Pitch.game_pk.in_(final_pks))
                .group_by(Pitch.game_pk)
            ).all()
        ) if final_pks else {}

    print(f"Coverage report: {start.isoformat()} -> {end.isoformat()}")
    print(f"  API total games:        {len(sched)}")
    print(f"  API played-Final games: {len(final_pks)}")
    for state, pks in skipped_detailed.items():
        print(f"  API {state}: {len(pks)}")
    for state, pks in nonfinal_by_state.items():
        print(f"  API {state}: {len(pks)}")
    print(f"  DB games in range:      {len(db_pks)}")

    missing_from_db = final_pks - db_pks
    print(f"  Final games missing from DB:       {len(missing_from_db)}")
    if missing_from_db:
        for pk in sorted(missing_from_db)[:20]:
            print(f"    {pk}")
        if len(missing_from_db) > 20:
            print(f"    ... and {len(missing_from_db) - 20} more")

    no_pitches = [pk for pk in final_pks if pk in db_pks and pitch_counts.get(pk, 0) == 0]
    print(f"  Final games in DB but with 0 pitches: {len(no_pitches)}")
    if no_pitches:
        for pk in sorted(no_pitches)[:20]:
            print(f"    {pk}")

    stale_status = [
        pk for pk, st in db_status_by_pk.items()
        if pk in final_pks and st != "Final"
    ]
    print(f"  DB rows with non-Final status that should be Final: {len(stale_status)}")

    print()
    return 0 if not missing_from_db and not no_pitches else 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    start = date.fromisoformat(sys.argv[1])
    end = date.fromisoformat(sys.argv[2])
    sys.exit(asyncio.run(main(start, end)))
