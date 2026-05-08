"""CLI entry points for the MLB pipeline."""

import asyncio

import click
import structlog

structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
)

logger = structlog.get_logger()


@click.group()
def main():
    """MLB Real-Time Data Pipeline CLI."""
    pass


@main.group()
def ingest():
    """Data ingestion commands."""
    pass


@ingest.command("live")
@click.option("--date", "target_date", default=None, help="Date to poll (YYYY-MM-DD, default: today)")
@click.option("--no-redis", is_flag=True, help="Disable Redis publishing")
def ingest_live(target_date: str | None, no_redis: bool):
    """Start live game poller for real-time data ingestion."""

    async def _run():
        from datetime import date as dt_date

        from sqlalchemy import create_engine

        from mlb_pipeline.config import settings
        from mlb_pipeline.ingestion.client import MLBStatsClient
        from mlb_pipeline.ingestion.poller import LiveGamePoller
        from mlb_pipeline.storage.sqlserver import SQLServerStorage
        from mlb_pipeline.stream.publisher import RedisPublisher

        poll_date = dt_date.fromisoformat(target_date) if target_date else dt_date.today()

        engine = create_engine(settings.db_connection_string)
        storage = SQLServerStorage(engine)

        publisher = None
        if not no_redis and settings.redis_enabled:
            publisher = RedisPublisher.create(
                settings.redis_url, maxlen=settings.redis_stream_maxlen
            )
            if publisher is None:
                click.echo("Warning: Redis unavailable, continuing without streaming.")

        click.echo(f"Starting live poller for {poll_date.isoformat()}...")

        async with MLBStatsClient() as client:
            poller = LiveGamePoller(
                client=client,
                storage=storage,
                publisher=publisher,
                settings=settings,
            )
            try:
                await poller.run(target_date=poll_date)
            finally:
                if publisher:
                    publisher.close()

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        click.echo("\nShutdown requested. Exiting cleanly.")


@ingest.command("backfill")
@click.option("--start", required=True, help="Start date (YYYY-MM-DD)")
@click.option("--end", required=True, help="End date (YYYY-MM-DD)")
@click.option("--concurrency", default=8, help="Concurrent live-feed fetches")
@click.option("--force", is_flag=True, help="Re-ingest games already present in DB")
def ingest_backfill(start: str, end: str, concurrency: int, force: bool):
    """Load historical Final games for a date range into SQL Server."""

    async def _run():
        from datetime import date as dt_date

        from sqlalchemy import create_engine

        from mlb_pipeline.config import settings
        from mlb_pipeline.ingestion.client import MLBStatsClient
        from mlb_pipeline.ingestion.historical import HistoricalGameLoader
        from mlb_pipeline.storage.sqlserver import SQLServerStorage

        start_date = dt_date.fromisoformat(start)
        end_date = dt_date.fromisoformat(end)

        engine = create_engine(settings.db_connection_string)
        storage = SQLServerStorage(engine)

        async with MLBStatsClient() as client:
            loader = HistoricalGameLoader(
                client=client, storage=storage, concurrency=concurrency
            )
            stats = await loader.backfill(
                start=start_date, end=end_date, force=force
            )

        click.echo("")
        click.echo(f"Backfill {start} -> {end}:")
        click.echo(f"  games found:     {stats.games_found}")
        click.echo(f"  games processed: {stats.games_processed}")
        click.echo(f"  games skipped:   {stats.games_skipped} (already in DB)")
        click.echo(f"  not yet final:   {stats.games_not_final}")
        click.echo(f"  failed:          {stats.games_failed}")
        click.echo(f"  pitches stored:  {stats.pitches_stored}")
        click.echo(f"  at-bats stored:  {stats.at_bats_stored}")
        if stats.failed_game_pks:
            click.echo(f"  failed game_pks: {stats.failed_game_pks}")

    asyncio.run(_run())


@ingest.command("game-metadata")
@click.option("--concurrency", default=12, help="Concurrent feed fetches")
@click.option(
    "--all", "all_games", is_flag=True,
    help="Refresh all games (default: only games missing venue_id or first_pitch)",
)
def ingest_game_metadata(concurrency: int, all_games: bool):
    """Backfill venue_id and first_pitch_time_utc on game rows."""

    async def _run():
        from sqlalchemy import create_engine

        from mlb_pipeline.config import settings
        from mlb_pipeline.ingestion.game_metadata import backfill_game_metadata

        engine = create_engine(settings.db_connection_string)
        stats = await backfill_game_metadata(
            engine, concurrency=concurrency, only_missing=not all_games
        )
        click.echo(f"Game metadata backfill:")
        click.echo(f"  processed:        {stats.games_processed}")
        click.echo(f"  venue updated:    {stats.venue_updated}")
        click.echo(f"  first pitch set:  {stats.first_pitch_updated}")
        click.echo(f"  failed:           {stats.failed}")

    asyncio.run(_run())


@ingest.command("venues")
@click.option("--concurrency", default=8, help="Concurrent venue fetches")
def ingest_venues_cmd(concurrency: int):
    """Fetch venue location and roof info for every venue referenced by games."""

    async def _run():
        from sqlalchemy import create_engine

        from mlb_pipeline.config import settings
        from mlb_pipeline.ingestion.client import MLBStatsClient
        from mlb_pipeline.ingestion.game_metadata import venue_ids_referenced_by_games
        from mlb_pipeline.ingestion.venues import ingest_venues

        engine = create_engine(settings.db_connection_string)
        venue_ids = venue_ids_referenced_by_games(engine)
        if not venue_ids:
            click.echo(
                "No venue_ids on games yet. Run 'mlb ingest game-metadata' first."
            )
            return

        async with MLBStatsClient() as client:
            saved = await ingest_venues(client, engine, venue_ids)

        click.echo(f"Venues ingested: {saved} of {len(venue_ids)}")

    asyncio.run(_run())


@ingest.command("weather")
@click.option("--start", default=None, help="Filter by game_date start (YYYY-MM-DD)")
@click.option("--end", default=None, help="Filter by game_date end (YYYY-MM-DD)")
@click.option("--concurrency", default=8, help="Concurrent Open-Meteo fetches")
def ingest_weather(start: str | None, end: str | None, concurrency: int):
    """Fetch first-pitch weather from Open-Meteo and store per-game."""

    async def _run():
        from datetime import date as dt_date

        from sqlalchemy import create_engine, select
        from sqlalchemy.orm import Session

        from mlb_pipeline.config import settings
        from mlb_pipeline.ingestion.weather import backfill_weather
        from mlb_pipeline.models.database import Game

        engine = create_engine(settings.db_connection_string)

        game_pks: list[int] | None = None
        if start or end:
            with Session(engine) as session:
                q = select(Game.game_pk).where(Game.status == "Final")
                if start:
                    q = q.where(Game.game_date >= dt_date.fromisoformat(start))
                if end:
                    q = q.where(Game.game_date <= dt_date.fromisoformat(end))
                game_pks = [r[0] for r in session.execute(q).all()]

        stats = await backfill_weather(
            engine, game_pks=game_pks, concurrency=concurrency
        )
        click.echo("Weather backfill:")
        click.echo(f"  games considered:   {stats.games_total}")
        click.echo(f"  fetched outdoor:    {stats.fetched}")
        click.echo(f"  indoor (skipped):   {stats.indoor_skipped}")
        click.echo(f"  no venue:           {stats.no_venue}")
        click.echo(f"  no first pitch:     {stats.no_first_pitch}")
        click.echo(f"  failed:             {stats.failed}")
        if stats.failed_game_pks:
            click.echo(f"  failed game_pks:    {stats.failed_game_pks[:20]}")

    asyncio.run(_run())


@ingest.command("weather-conditions")
@click.option("--concurrency", default=12, help="Concurrent feed fetches")
def ingest_weather_conditions(concurrency: int):
    """Enrich game_weather rows with MLB feed condition / roof state."""

    async def _run():
        from sqlalchemy import create_engine

        from mlb_pipeline.config import settings
        from mlb_pipeline.ingestion.weather import enrich_with_mlb_conditions

        engine = create_engine(settings.db_connection_string)
        n = await enrich_with_mlb_conditions(engine, concurrency=concurrency)
        click.echo(f"Enriched {n} weather rows with MLB feed conditions.")

    asyncio.run(_run())


@main.command("replay")
@click.argument("game_pk", type=int)
@click.option("--speed", default=10.0, help="Replay speed multiplier")
def replay(game_pk: int, speed: float):
    """Replay a completed game through the pipeline."""
    click.echo(f"Replaying game {game_pk} at {speed}x speed... (Phase 3)")
    # Will be implemented in Phase 3


@main.group()
def db():
    """Database management commands."""
    pass


@db.command("migrate")
def db_migrate():
    """Run database migrations."""
    from alembic import command
    from alembic.config import Config

    alembic_cfg = Config("alembic.ini")
    command.upgrade(alembic_cfg, "head")
    click.echo("Migrations applied successfully.")


@db.command("seed-teams")
def db_seed_teams():
    """Seed teams and players reference data from MLB API."""

    async def _seed():
        from sqlalchemy import create_engine
        from sqlalchemy.orm import Session

        from mlb_pipeline.config import settings
        from mlb_pipeline.ingestion.client import MLBStatsClient
        from mlb_pipeline.models.database import Base, Team

        engine = create_engine(settings.db_connection_string)
        async with MLBStatsClient() as client:
            data = await client.get_teams(season=2025)
            teams = data.get("teams", [])

        with Session(engine) as session:
            for t in teams:
                team = Team(
                    team_id=t["id"],
                    name=t["name"],
                    abbreviation=t.get("abbreviation", ""),
                    league=t.get("league", {}).get("name", ""),
                    division=t.get("division", {}).get("name", ""),
                    venue_name=t.get("venue", {}).get("name"),
                )
                session.merge(team)
            session.commit()
            click.echo(f"Seeded {len(teams)} teams.")

    asyncio.run(_seed())


@db.command("test-connection")
def db_test_connection():
    """Test SQL Server connectivity."""
    from sqlalchemy import create_engine, text

    from mlb_pipeline.config import settings

    engine = create_engine(settings.db_connection_string)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT @@VERSION")).scalar()
        click.echo(f"Connected to: {result[:80]}...")


@main.command("api-test")
@click.option("--date", default=None, help="Date to query (YYYY-MM-DD)")
def api_test(date: str | None):
    """Test MLB Stats API connectivity."""

    async def _test():
        from datetime import date as dt_date

        from mlb_pipeline.ingestion.client import MLBStatsClient

        if date is None:
            query_date = dt_date.today().isoformat()
        else:
            query_date = date

        async with MLBStatsClient() as client:
            schedule = await client.get_schedule(query_date)
            dates = schedule.get("dates", [])
            if not dates:
                click.echo(f"No games scheduled for {query_date}")
                return
            games = dates[0].get("games", [])
            click.echo(f"Found {len(games)} games for {query_date}:")
            for g in games:
                away = g["teams"]["away"]["team"]["name"]
                home = g["teams"]["home"]["team"]["name"]
                status = g["status"]["detailedState"]
                click.echo(f"  {g['gamePk']}: {away} @ {home} — {status}")

    asyncio.run(_test())


@main.group()
def ml():
    """Machine learning commands."""
    pass


@ml.command("train")
@click.option("--data", required=True, help="Path to training data Parquet")
@click.option("--epochs", default=30, help="Number of training epochs")
def ml_train(data: str, epochs: int):
    """Train win probability model."""
    click.echo(f"Training model on {data} for {epochs} epochs... (Phase 4)")


@ml.command("evaluate")
@click.option("--model", required=True, help="Path to model checkpoint")
def ml_evaluate(model: str):
    """Evaluate model performance."""
    click.echo(f"Evaluating model {model}... (Phase 4)")


@main.command("dashboard")
@click.option("--port", default=8000, help="API server port")
def dashboard(port: int):
    """Start the FastAPI dashboard server."""
    click.echo(f"Starting dashboard API on port {port}... (Phase 6)")


if __name__ == "__main__":
    main()
