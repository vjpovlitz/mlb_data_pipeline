"""DAG: Daily backfill — pull completed games from MLB API and store to SQL Server.

Runs daily at 8 AM ET (after West Coast games finish). Fetches yesterday's
completed games, parses all pitches and at-bats, and stores them to SQL Server.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator

default_args = {
    "owner": "mlb_pipeline",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "daily_backfill",
    default_args=default_args,
    description="Backfill completed games from MLB API into SQL Server",
    schedule="0 12 * * *",  # 8 AM ET = 12:00 UTC
    start_date=datetime(2026, 3, 27),
    catchup=False,
    tags=["mlb", "ingestion"],
)


def fetch_and_store_games(**context):
    """Fetch yesterday's completed games and store all events."""
    import asyncio

    from sqlalchemy import create_engine

    from mlb_pipeline.config import settings
    from mlb_pipeline.ingestion.client import MLBStatsClient
    from mlb_pipeline.ingestion.parser import parse_all_game_events
    from mlb_pipeline.storage.sqlserver import SQLServerStorage

    execution_date = context["ds"]  # YYYY-MM-DD string

    engine = create_engine(settings.db_connection_string)
    storage = SQLServerStorage(engine)

    async def _fetch():
        async with MLBStatsClient() as client:
            schedule = await client.get_schedule(execution_date)
            dates = schedule.get("dates", [])
            if not dates:
                print(f"No games found for {execution_date}")
                return {"games": 0, "pitches": 0, "at_bats": 0}

            games = dates[0].get("games", [])
            total_pitches = 0
            total_at_bats = 0
            games_processed = 0

            for game_data in games:
                game_pk = game_data["gamePk"]
                status = game_data["status"]["abstractGameState"]

                if status != "Final":
                    continue

                teams = game_data["teams"]
                storage.upsert_game(
                    game_pk=game_pk,
                    game_date=datetime.fromisoformat(execution_date).date(),
                    game_type=game_data.get("gameType", "R"),
                    season=int(game_data.get("season", 2026)),
                    status=status,
                    away_team_id=teams["away"]["team"]["id"],
                    away_team_name=teams["away"]["team"]["name"],
                    home_team_id=teams["home"]["team"]["id"],
                    home_team_name=teams["home"]["team"]["name"],
                    venue_name=game_data.get("venue", {}).get("name"),
                )

                feed = await client.get_live_feed(game_pk)
                pitches, at_bats = parse_all_game_events(game_pk, feed)

                p_count = storage.insert_pitches_batch(pitches)
                ab_count = storage.insert_at_bats_batch(at_bats)

                linescore = feed.get("liveData", {}).get("linescore", {})
                teams_score = linescore.get("teams", {})
                away_score = teams_score.get("away", {}).get("runs", 0) or 0
                home_score = teams_score.get("home", {}).get("runs", 0) or 0
                storage.update_game_score(game_pk, away_score, home_score, status)

                total_pitches += p_count
                total_at_bats += ab_count
                games_processed += 1

                print(f"  {game_pk}: {teams['away']['team']['name']} @ "
                      f"{teams['home']['team']['name']} — {p_count} pitches, {ab_count} ABs")

            return {
                "games": games_processed,
                "pitches": total_pitches,
                "at_bats": total_at_bats,
            }

    result = asyncio.run(_fetch())
    print(f"Backfill complete for {execution_date}: {result}")
    return result


def export_to_parquet(**context):
    """Export the day's data to Parquet for analytics."""
    from datetime import date as dt_date
    from pathlib import Path

    from sqlalchemy import create_engine

    from mlb_pipeline.config import settings
    from mlb_pipeline.storage.parquet import ParquetStore

    execution_date = dt_date.fromisoformat(context["ds"])
    engine = create_engine(settings.db_connection_string)
    store = ParquetStore(settings.parquet_dir)

    result = store.export_date_range(engine, execution_date, execution_date)
    print(f"Exported to Parquet: {result}")
    return result


fetch_task = PythonOperator(
    task_id="fetch_and_store_games",
    python_callable=fetch_and_store_games,
    dag=dag,
)

export_task = PythonOperator(
    task_id="export_to_parquet",
    python_callable=export_to_parquet,
    dag=dag,
)

fetch_task >> export_task
