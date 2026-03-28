"""Enumerations for MLB game data."""

from enum import StrEnum


class GameState(StrEnum):
    PREVIEW = "Preview"
    LIVE = "Live"
    FINAL = "Final"
    POSTPONED = "Postponed"
    SUSPENDED = "Suspended"


class HalfInning(StrEnum):
    TOP = "top"
    BOTTOM = "bottom"


class PitchType(StrEnum):
    FF = "FF"  # 4-Seam Fastball
    SI = "SI"  # Sinker
    FC = "FC"  # Cutter
    SL = "SL"  # Slider
    CU = "CU"  # Curveball
    CH = "CH"  # Changeup
    FS = "FS"  # Splitter
    KC = "KC"  # Knuckle Curve
    KN = "KN"  # Knuckleball
    EP = "EP"  # Eephus
    ST = "ST"  # Sweeper
    SV = "SV"  # Slurve
    SC = "SC"  # Screwball
    CS = "CS"  # Slow Curve
    FA = "FA"  # Fastball (generic)
    PO = "PO"  # Pitchout
    IN = "IN"  # Intentional Ball
    AB = "AB"  # Auto Ball


class EventType(StrEnum):
    PITCH = "pitch"
    PICKOFF = "pickoff"
    STOLEN_BASE = "stolen_base"
    CAUGHT_STEALING = "caught_stealing"
    WILD_PITCH = "wild_pitch"
    PASSED_BALL = "passed_ball"
    BALK = "balk"
    SUBSTITUTION = "substitution"
    GAME_ADVISORY = "game_advisory"
    OTHER = "other"
