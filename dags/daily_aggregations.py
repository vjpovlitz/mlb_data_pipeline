"""DAG: Daily aggregations — compute player and team stats after games complete.

Runs after daily_backfill. Computes pitcher/batter game stats, updates rolling
player stats in player_stats_daily, and generates team-level summaries.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

default_args = {
    "owner": "mlb_pipeline",
    "depends_on_past": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "daily_aggregations",
    default_args=default_args,
    description="Compute daily player/team stats from game data",
    schedule="0 13 * * *",  # 1 hour after backfill
    start_date=datetime(2026, 3, 27),
    catchup=False,
    tags=["mlb", "analytics"],
)


def compute_pitcher_stats(**context):
    """Aggregate pitcher stats for the day's games."""
    import polars as pl
    from sqlalchemy import create_engine, text

    from mlb_pipeline.config import settings
    from mlb_pipeline.processing.aggregations import aggregate_pitcher_game

    execution_date = context["ds"]
    engine = create_engine(settings.db_connection_string)

    with engine.connect() as conn:
        pitches_df = pl.read_database(
            f"SELECT p.* FROM pitches p JOIN games g ON p.game_pk = g.game_pk "
            f"WHERE g.game_date = '{execution_date}' AND g.status = 'Final'",
            connection=conn,
        )

    if pitches_df.is_empty():
        print(f"No pitch data for {execution_date}")
        return {}

    pitcher_stats = aggregate_pitcher_game(pitches_df.lazy()).collect()
    print(f"Computed stats for {pitcher_stats.height} pitchers across "
          f"{pitcher_stats['game_pk'].n_unique()} games")
    return {"pitchers": pitcher_stats.height}


def compute_batter_stats(**context):
    """Aggregate batter stats for the day's games."""
    import polars as pl
    from sqlalchemy import create_engine

    from mlb_pipeline.config import settings
    from mlb_pipeline.processing.aggregations import aggregate_batter_game

    execution_date = context["ds"]
    engine = create_engine(settings.db_connection_string)

    with engine.connect() as conn:
        at_bats_df = pl.read_database(
            f"SELECT ab.* FROM at_bats ab JOIN games g ON ab.game_pk = g.game_pk "
            f"WHERE g.game_date = '{execution_date}' AND g.status = 'Final'",
            connection=conn,
        )
        pitches_df = pl.read_database(
            f"SELECT p.* FROM pitches p JOIN games g ON p.game_pk = g.game_pk "
            f"WHERE g.game_date = '{execution_date}' AND g.status = 'Final'",
            connection=conn,
        )

    if at_bats_df.is_empty():
        print(f"No at-bat data for {execution_date}")
        return {}

    batter_stats = aggregate_batter_game(at_bats_df.lazy(), pitches_df.lazy()).collect()
    print(f"Computed stats for {batter_stats.height} batters")
    return {"batters": batter_stats.height}


def update_daily_player_stats(**context):
    """Write aggregated stats to player_stats_daily table."""
    from datetime import date as dt_date

    import polars as pl
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import Session

    from mlb_pipeline.config import settings
    from mlb_pipeline.models.database import PlayerStatsDaily

    execution_date = dt_date.fromisoformat(context["ds"])
    engine = create_engine(settings.db_connection_string)

    # Query batter stats from at_bats
    with engine.connect() as conn:
        at_bats_df = pl.read_database(
            f"SELECT ab.* FROM at_bats ab JOIN games g ON ab.game_pk = g.game_pk "
            f"WHERE g.game_date = '{execution_date}' AND g.status = 'Final'",
            connection=conn,
        )
        pitches_df = pl.read_database(
            f"SELECT p.* FROM pitches p JOIN games g ON p.game_pk = g.game_pk "
            f"WHERE g.game_date = '{execution_date}' AND g.status = 'Final'",
            connection=conn,
        )

    if at_bats_df.is_empty():
        print(f"No data for {execution_date}")
        return 0

    # Compute batter daily stats
    batter_agg = (
        at_bats_df.lazy()
        .group_by("batter_id")
        .agg(
            pl.len().alias("plate_appearances"),
            pl.col("event_type")
            .filter(~pl.col("event_type").is_in(["walk", "hit_by_pitch", "sac_fly", "sac_bunt"]))
            .len()
            .alias("at_bats"),
            pl.col("event_type")
            .filter(pl.col("event_type").is_in(["single", "double", "triple", "home_run"]))
            .len()
            .alias("hits"),
            pl.col("event_type").filter(pl.col("event_type") == "double").len().alias("doubles"),
            pl.col("event_type").filter(pl.col("event_type") == "triple").len().alias("triples"),
            pl.col("event_type").filter(pl.col("event_type") == "home_run").len().alias("home_runs"),
            pl.col("rbi").sum().alias("rbi"),
            pl.col("event_type").filter(pl.col("event_type") == "walk").len().alias("walks"),
            pl.col("event_type").filter(pl.col("event_type") == "strikeout").len().alias("strikeouts"),
            pl.col("game_pk").n_unique().alias("games"),
        )
        .collect()
    )

    # Compute pitcher daily stats
    pitcher_agg = (
        pitches_df.lazy()
        .group_by("pitcher_id")
        .agg(
            pl.len().alias("pitches_thrown"),
            pl.col("is_strike").sum().alias("strikeouts_pitching_approx"),
        )
        .collect()
    )

    # Write to DB
    count = 0
    with Session(engine) as session:
        for row in batter_agg.iter_rows(named=True):
            stat = PlayerStatsDaily(
                player_id=row["batter_id"],
                stat_date=execution_date,
                games=row["games"],
                plate_appearances=row["plate_appearances"],
                at_bats=row["at_bats"],
                hits=row["hits"],
                doubles=row["doubles"],
                triples=row["triples"],
                home_runs=row["home_runs"],
                rbi=row["rbi"],
                walks=row["walks"],
                strikeouts=row["strikeouts"],
            )
            session.merge(stat)
            count += 1
        session.commit()

    print(f"Updated {count} player daily stat records for {execution_date}")
    return count


wait_for_backfill = ExternalTaskSensor(
    task_id="wait_for_backfill",
    external_dag_id="daily_backfill",
    external_task_id="export_to_parquet",
    timeout=3600,
    poke_interval=60,
    dag=dag,
)

pitcher_task = PythonOperator(
    task_id="compute_pitcher_stats",
    python_callable=compute_pitcher_stats,
    dag=dag,
)

batter_task = PythonOperator(
    task_id="compute_batter_stats",
    python_callable=compute_batter_stats,
    dag=dag,
)

daily_stats_task = PythonOperator(
    task_id="update_daily_player_stats",
    python_callable=update_daily_player_stats,
    dag=dag,
)

wait_for_backfill >> [pitcher_task, batter_task] >> daily_stats_task
