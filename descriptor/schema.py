"""Telemetry schema for real-time camera feed descriptions (~1 Hz).

Emitted once per second for each camera stream in a multi-camera match setup:
Allows the AI Director policy to select the best broadcast camera feed deterministically.
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple, Any
import json


@dataclass
class CameraFeedTelemetry:
    """Standardized ~1 Hz per-camera telemetry descriptor."""

    camera_id: str
    timestamp: float
    frame_index: int

    # 1. Shot / Scene classification
    shot_type: str             # "WIDE_TACTICAL", "MEDIUM_PLAY", "TIGHT_CLOSEUP", "CROWD_BENCH", "REPLAY"
    shot_confidence: float
    camera_motion: str         # "STATIONARY", "SMOOTH_PAN", "RAPID_PAN", "CAMERA_CUT"

    # 2. Ball state (powered by HighRecallBallTracker)
    ball_visible: bool
    ball_coords_norm: Optional[Tuple[float, float]]   # (x/w, y/h)
    ball_speed_px_f: float
    ball_trajectory_status: str                       # "GROUND_PLAY", "AIRBORNE_SHOT", "LOST"

    # 3. Match tempo & kinematics
    play_tempo: str            # "HIGH_COUNTER", "ATTACKING_BUILDUP", "SET_PIECE", "STOPPAGE"
    attacking_speed_kmh: Optional[float]

    # 4. Entities in frame
    player_count: int
    ball_carrier_id: Optional[str]
    is_referee_in_focus: bool
    team_in_possession: Optional[int]

    # 5. Live match events (active triggers)
    event_type: Optional[str]  # "GOAL", "SHOT_ON_TARGET", "FOUL_TACKLE", "CELEBRATION", None
    event_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
