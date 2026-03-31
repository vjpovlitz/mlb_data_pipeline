"""DAG: Data quality — validate pitch data, check for gaps and anomalies.

Runs after daily_aggregations. Validates pitch metrics are within expected ranges,
checks for duplicate events, and flags missing data.
"""

from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.sensors.external_task import ExternalTaskSensor

default_args = {
    "owner": "mlb_pipeline",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

dag = DAG(
    "data_quality",
    default_args=default_args,
    description="Validate data quality for ingested game data",
    schedule="0 14 * * *",  # After aggregations
    start_date=datetime(2026, 3, 27),
    catchup=False,
    tags=["mlb", "quality"],
)


def validate_pitch_ranges(**context):
    """Check that pitch metrics fall within expected physical ranges."""
    import polars as pl
    from sqlalchemy import create_engine

    from mlb_pipeline.config import settings

    execution_date = context["ds"]
    engine = create_engine(settings.db_connection_string)

    with engine.connect() as conn:
        df = pl.read_database(
            f"SELECT p.* FROM pitches p JOIN games g ON p.game_pk = g.game_pk "
            f"WHERE g.game_date = '{execution_date}'",
            connection=conn,
        )

    if df.is_empty():
        print(f"No pitches to validate for {execution_date}")
        return {"status": "skip", "reason": "no data"}

    issues = []

    # Speed range: 40-110 mph (allowing for eephus and extreme fastballs)
    speed_violations = df.filter(
        (pl.col("start_speed").is_not_null())
        & ((pl.col("start_speed") < 40) | (pl.col("start_speed") > 110))
    )
    if speed_violations.height > 0:
        issues.append(f"Speed out of range (40-110 mph): {speed_violations.height} pitches")

    # Plate location: plate_x ±3.0 ft, plate_z -1.0 to 6.0 ft
    location_violations = df.filter(
        (pl.col("plate_x").is_not_null())
        & ((pl.col("plate_x").abs() > 3.0) | (pl.col("plate_z") < -1.0) | (pl.col("plate_z") > 6.0))
    )
    if location_violations.height > 0:
        issues.append(f"Plate location out of range: {location_violations.height} pitches")

    # Spin rate: 0-4000 rpm
    spin_violations = df.filter(
        (pl.col("spin_rate").is_not_null())
        & ((pl.col("spin_rate") < 0) | (pl.col("spin_rate") > 4000))
    )
    if spin_violations.height > 0:
        issues.append(f"Spin rate out of range (0-4000): {spin_violations.height} pitches")

    # NULL rate for key fields should be < 10%
    total = df.height
    for col in ["pitch_type", "start_speed", "plate_x", "plate_z"]:
        null_count = df.filter(pl.col(col).is_null()).height
        null_pct = null_count / total * 100
        if null_pct > 10:
            issues.append(f"High NULL rate for {col}: {null_pct:.1f}% ({null_count}/{total})")

    result = {
        "date": execution_date,
        "total_pitches": total,
        "issues": issues,
        "status": "fail" if issues else "pass",
    }

    for issue in issues:
        print(f"  WARNING: {issue}")

    if not issues:
        print(f"All {total} pitches pass quality checks for {execution_date}")

    return result


def check_duplicates(**context):
    """Check for duplicate pitch events."""
    import polars as pl
    from sqlalchemy import create_engine

    from mlb_pipeline.config import settings

    execution_date = context["ds"]
    engine = create_engine(settings.db_connection_string)

    with engine.connect() as conn:
        df = pl.read_database(
            f"SELECT event_id, COUNT(*) as cnt FROM pitches p "
            f"JOIN games g ON p.game_pk = g.game_pk "
            f"WHERE g.game_date = '{execution_date}' "
            f"GROUP BY event_id HAVING COUNT(*) > 1",
            connection=conn,
        )

    if df.height > 0:
        print(f"WARNING: {df.height} duplicate event_ids found for {execution_date}")
        return {"status": "fail", "duplicates": df.height}

    print(f"No duplicates found for {execution_date}")
    return {"status": "pass", "duplicates": 0}


def check_game_completeness(**context):
    """Verify all Final games have pitches and at-bats stored."""
    import polars as pl
    from sqlalchemy import create_engine

    from mlb_pipeline.config import settings

    execution_date = context["ds"]
    engine = create_engine(settings.db_connection_string)

    with engine.connect() as conn:
        games_df = pl.read_database(
            f"SELECT game_pk, away_team_name, home_team_name FROM games "
            f"WHERE game_date = '{execution_date}' AND status = 'Final'",
            connection=conn,
        )

        if games_df.is_empty():
            print(f"No Final games for {execution_date}")
            return {"status": "skip"}

        pitch_counts = pl.read_database(
            f"SELECT game_pk, COUNT(*) as pitch_count FROM pitches "
            f"WHERE game_pk IN ({','.join(str(pk) for pk in games_df['game_pk'].to_list())}) "
            f"GROUP BY game_pk",
            connection=conn,
        )

        ab_counts = pl.read_database(
            f"SELECT game_pk, COUNT(*) as ab_count FROM at_bats "
            f"WHERE game_pk IN ({','.join(str(pk) for pk in games_df['game_pk'].to_list())}) "
            f"GROUP BY game_pk",
            connection=conn,
        )

    issues = []
    for row in games_df.iter_rows(named=True):
        game_pk = row["game_pk"]
        matchup = f"{row['away_team_name']} @ {row['home_team_name']}"

        p_count = pitch_counts.filter(pl.col("game_pk") == game_pk)
        pitches = p_count["pitch_count"][0] if p_count.height > 0 else 0

        ab_count = ab_counts.filter(pl.col("game_pk") == game_pk)
        at_bats = ab_count["ab_count"][0] if ab_count.height > 0 else 0

        if pitches == 0:
            issues.append(f"{game_pk} ({matchup}): 0 pitches stored")
        elif pitches < 200:
            issues.append(f"{game_pk} ({matchup}): only {pitches} pitches (suspiciously low)")

        if at_bats == 0:
            issues.append(f"{game_pk} ({matchup}): 0 at-bats stored")

    for issue in issues:
        print(f"  WARNING: {issue}")

    if not issues:
        print(f"All {games_df.height} games complete for {execution_date}")

    return {
        "games_checked": games_df.height,
        "issues": issues,
        "status": "fail" if issues else "pass",
    }


wait_for_aggregations = ExternalTaskSensor(
    task_id="wait_for_aggregations",
    external_dag_id="daily_aggregations",
    external_task_id="update_daily_player_stats",
    timeout=3600,
    poke_interval=60,
    dag=dag,
)

validate_ranges = PythonOperator(
    task_id="validate_pitch_ranges",
    python_callable=validate_pitch_ranges,
    dag=dag,
)

check_dupes = PythonOperator(
    task_id="check_duplicates",
    python_callable=check_duplicates,
    dag=dag,
)

check_complete = PythonOperator(
    task_id="check_game_completeness",
    python_callable=check_game_completeness,
    dag=dag,
)

wait_for_aggregations >> [validate_ranges, check_dupes, check_complete]
