"""Parse MLB Stats API live feed responses into pipeline events.

This module converts raw MLB API JSON into canonical PitchEvent and AtBatResult
objects. It is used by both the live poller (incrementally) and the historical
loader (full game at once).
"""

from datetime import datetime, timezone

from mlb_pipeline.models.enums import HalfInning
from mlb_pipeline.models.events import AtBatResult, PitchEvent


def parse_pitch_events(game_pk: int, play: dict) -> list[PitchEvent]:
    """Extract all pitch events from a single at-bat (play) entry.

    Filters out non-pitch events (pickoffs, mound visits, etc.) and returns
    only actual pitches with their metrics.
    """
    about = play["about"]
    matchup = play["matchup"]
    result = play["result"]

    pitcher = matchup["pitcher"]
    batter = matchup["batter"]
    half = HalfInning.TOP if about["isTopInning"] else HalfInning.BOTTOM

    pitches: list[PitchEvent] = []
    pitch_number = 0

    for event in play.get("playEvents", []):
        if not event.get("isPitch", False):
            continue

        pitch_number += 1
        details = event.get("details", {})
        pitch_data = event.get("pitchData", {})
        hit_data = event.get("hitData", {})
        count = event.get("count", {})
        breaks = pitch_data.get("breaks", {})
        coordinates = pitch_data.get("coordinates", {})

        # Parse call info
        call = details.get("call", {})
        call_code = call.get("code", "")
        call_desc = call.get("description", "")
        pitch_type_info = details.get("type", {})
        pitch_type_code = pitch_type_info.get("code") if pitch_type_info else None

        is_in_play = details.get("isInPlay", False)
        is_strike = details.get("isStrike", False)
        is_ball = details.get("isBall", False)

        # Parse timestamp
        start_time = event.get("startTime") or event.get("endTime") or about.get("startTime")
        if start_time:
            ts = datetime.fromisoformat(start_time.replace("Z", "+00:00"))
        else:
            ts = datetime.now(timezone.utc)

        pitch = PitchEvent(
            game_pk=game_pk,
            event_id=f"{game_pk}_{about['atBatIndex']}_{pitch_number}",
            timestamp=ts,
            inning=about["inning"],
            half_inning=half,
            at_bat_index=about["atBatIndex"],
            pitch_number=pitch_number,
            pitcher_id=pitcher["id"],
            pitcher_name=pitcher["fullName"],
            batter_id=batter["id"],
            batter_name=batter["fullName"],
            pitch_type=pitch_type_code,
            start_speed=pitch_data.get("startSpeed"),
            end_speed=pitch_data.get("endSpeed"),
            spin_rate=breaks.get("spinRate"),
            spin_direction=breaks.get("spinDirection"),
            break_angle=breaks.get("breakAngle"),
            break_length=breaks.get("breakLength"),
            pfx_x=coordinates.get("pfxX"),
            pfx_z=coordinates.get("pfxZ"),
            plate_x=coordinates.get("pX"),
            plate_z=coordinates.get("pZ"),
            zone=pitch_data.get("zone"),
            call_code=call_code,
            call_description=call_desc,
            is_in_play=is_in_play,
            is_strike=is_strike,
            is_ball=is_ball,
            launch_speed=hit_data.get("launchSpeed") if is_in_play else None,
            launch_angle=hit_data.get("launchAngle") if is_in_play else None,
            total_distance=hit_data.get("totalDistance") if is_in_play else None,
            trajectory=hit_data.get("trajectory") if is_in_play else None,
            balls=count.get("balls", 0),
            strikes=count.get("strikes", 0),
            outs=count.get("outs", 0),
            away_score=result.get("awayScore", 0),
            home_score=result.get("homeScore", 0),
        )
        pitches.append(pitch)

    return pitches


def parse_at_bat_result(game_pk: int, play: dict) -> AtBatResult:
    """Parse a completed at-bat into an AtBatResult."""
    about = play["about"]
    matchup = play["matchup"]
    result = play["result"]
    half = HalfInning.TOP if about["isTopInning"] else HalfInning.BOTTOM

    # Count actual pitches
    pitch_count = sum(1 for e in play.get("playEvents", []) if e.get("isPitch", False))

    # Parse timestamp
    end_time = about.get("endTime") or about.get("startTime")
    if end_time:
        ts = datetime.fromisoformat(end_time.replace("Z", "+00:00"))
    else:
        ts = datetime.now(timezone.utc)

    # Determine runner state from the last event's count or default
    last_count = {"outs": 0}
    for event in reversed(play.get("playEvents", [])):
        if "count" in event:
            last_count = event["count"]
            break

    return AtBatResult(
        game_pk=game_pk,
        at_bat_index=about["atBatIndex"],
        timestamp=ts,
        inning=about["inning"],
        half_inning=half,
        pitcher_id=matchup["pitcher"]["id"],
        pitcher_name=matchup["pitcher"]["fullName"],
        batter_id=matchup["batter"]["id"],
        batter_name=matchup["batter"]["fullName"],
        event=result.get("event", "Unknown"),
        event_type=result.get("eventType", "unknown"),
        description=result.get("description", ""),
        rbi=result.get("rbi", 0),
        away_score=result.get("awayScore", 0),
        home_score=result.get("homeScore", 0),
        is_scoring_play=about.get("isScoringPlay", False),
        outs_after=last_count.get("outs", 0),
        pitch_count=pitch_count,
    )


def parse_all_game_events(
    game_pk: int, feed: dict
) -> tuple[list[PitchEvent], list[AtBatResult]]:
    """Parse an entire game's live feed into pitch events and at-bat results.

    Returns (pitches, at_bats) tuple.
    """
    all_pitches: list[PitchEvent] = []
    all_at_bats: list[AtBatResult] = []

    plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
    for play in plays:
        pitches = parse_pitch_events(game_pk, play)
        all_pitches.extend(pitches)
        at_bat = parse_at_bat_result(game_pk, play)
        all_at_bats.append(at_bat)

    return all_pitches, all_at_bats
