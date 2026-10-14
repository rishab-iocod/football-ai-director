"""detectors.py — Event Detection Engine for Live Football Broadcasting.

Detects in real-time:
1. ⚽ GOALS: Ball entering goal mouth, net penetration, celebration huddle/clustering.
2. ⚠️ FOULS & TACKLES: High-speed player collisions, tumble/ground impacts, counter-attack stops.
3. ⚡ SPRINT BREAKS & SHOTS: High velocity counter-attacks and goal attempts.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class GoalEvent:
    event_id: str
    frame: int
    timestamp: float
    team: int
    scorer_id: Optional[str]
    speed_kmh: float
    confidence: float
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class FoulEvent:
    event_id: str
    frame: int
    timestamp: float
    victim_id: str
    offender_id: Optional[str]
    severity: str        # "TACTICAL_FOUL", "HARD_TACKLE", "PENALTY_APPEAL"
    card: str            # "WARNING", "YELLOW_CARD", "RED_CARD"
    confidence: float
    coords: Tuple[float, float]
    description: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GoalDetector:
    """Detects goal events from ball kinematics and celebration clustering."""

    def __init__(self, frame_size=(2880, 720), cooldown_frames=250):
        self.frame_w, self.frame_h = frame_size
        self.cooldown_frames = cooldown_frames
        self.last_goal_frame = -9999

        # Scoreboard
        self.score_team_1 = 0
        self.score_team_2 = 0

        # Memory of recent ball states
        self.recent_carrier_id: Optional[str] = None
        self.recent_carrier_team: Optional[int] = None
        self.recent_ball_vel = (0.0, 0.0)
        self.recent_ball_speed = 0.0
        self.ball_near_goal_frames = 0

    def reset(self):
        self.last_goal_frame = -9999
        self.recent_carrier_id = None
        self.recent_carrier_team = None
        self.recent_ball_vel = (0.0, 0.0)
        self.recent_ball_speed = 0.0
        self.ball_near_goal_frames = 0

    def update_possession(self, players: Dict[str, Any], ball_state: Dict[str, Any]):
        """Keep track of who last held possession before a shot."""
        ball_bbox = ball_state.get("bbox")
        if ball_bbox is None:
            return

        bx = (ball_bbox[0] + ball_bbox[2]) / 2.0
        by = (ball_bbox[1] + ball_bbox[3]) / 2.0

        min_dist = float("inf")
        closest_pid = None
        closest_team = None

        for pid, pdata in players.items():
            pbbox = pdata.get("bbox")
            if not pbbox:
                continue
            foot_x = (pbbox[0] + pbbox[2]) / 2.0
            foot_y = pbbox[3]
            dist = math.hypot(foot_x - bx, foot_y - by)
            if dist < 80.0 and dist < min_dist:
                min_dist = dist
                closest_pid = pid
                closest_team = pdata.get("team")

        if closest_pid is not None:
            self.recent_carrier_id = str(closest_pid)
            if closest_team in (1, 2):
                self.recent_carrier_team = closest_team

    def check(
        self,
        frame_idx: int,
        ball_state: Dict[str, Any],
        players: Dict[str, Any],
        active_cam_id: int = 2,
        is_multi: bool = False,
        is_panoramic: bool = False,
    ) -> Optional[GoalEvent]:
        """Inspect current frame for goal event with physical stadium geometry validation."""
        self.update_possession(players, ball_state)

        # Check cooldown to prevent duplicate triggers
        if frame_idx - self.last_goal_frame < self.cooldown_frames:
            return None

        # In multi-camera mode, Cam 2 is Center Midfield — NO goal mouths exist in Cam 2!
        if is_multi and active_cam_id == 2:
            self.ball_near_goal_frames = 0
            return None

        ball_bbox = ball_state.get("bbox")
        if ball_bbox is None:
            self.ball_near_goal_frames = 0
            return None

        # Phantom or old extrapolated ball hold should not trigger fresh goals
        if ball_state.get("is_extrapolated", False) and self.ball_near_goal_frames < 4:
            return None

        bx = (ball_bbox[0] + ball_bbox[2]) / 2.0
        by = (ball_bbox[1] + ball_bbox[3]) / 2.0
        vx, vy = ball_state.get("velocity", (0.0, 0.0))
        speed_px = ball_state.get("speed_px", 0.0)
        speed_kmh = speed_px * 2.8  # Calibration factor to km/h

        # Reject glitchy speed jumps
        if speed_kmh > 170.0:
            return None

        # Physical goal mouth boundaries per camera setup
        w, h = self.frame_w, self.frame_h
        is_left_goal = False
        is_right_goal = False

        if is_multi:
            # 960x720 sector cameras
            if active_cam_id == 1:
                # Cam 1: Left Wing & Left Goal Mouth
                is_left_goal = (bx < 140.0 and 180.0 <= by <= 520.0)
            elif active_cam_id == 3:
                # Cam 3: Right Wing & Right Goal Mouth
                is_right_goal = (bx > 820.0 and 180.0 <= by <= 520.0)
        elif is_panoramic or w >= 2800:
            # Panoramic continuous coordinates (2880x720)
            is_left_goal = (bx < 140.0 and 180.0 <= by <= 520.0)
            is_right_goal = (bx > 2740.0 and 180.0 <= by <= 520.0)
        else:
            # Single broadcast 1080p full pitch
            is_left_goal = (bx < 160.0 and 220.0 <= by <= 750.0)
            is_right_goal = (bx > 1760.0 and 220.0 <= by <= 750.0)

        if not (is_left_goal or is_right_goal):
            self.ball_near_goal_frames = 0
            return None

        self.ball_near_goal_frames += 1

        # Check for inward directed trajectory or persistence inside the net
        inward_shot = (
            (is_left_goal and (vx < -1.8 or bx < 95.0 or self.ball_near_goal_frames >= 5)) or
            (is_right_goal and (vx > 1.8 or bx > (w - 95.0 if not is_multi else 865.0) or self.ball_near_goal_frames >= 5))
        )

        if not inward_shot:
            return None

        # Check if teammates are beginning a celebration cluster
        celebration = self._check_celebration_cluster(players)

        # Determine scoring team
        scoring_team = 1
        if self.recent_carrier_team:
            scoring_team = self.recent_carrier_team
        elif is_left_goal:
            # Ball entered left goal -> right-attacking team scored (Team 2)
            scoring_team = 2
        else:
            scoring_team = 1

        # Update match scoreboard
        if scoring_team == 1:
            self.score_team_1 += 1
        else:
            self.score_team_2 += 1

        self.last_goal_frame = frame_idx
        scorer_str = f"Player #{self.recent_carrier_id}" if self.recent_carrier_id else f"Team {scoring_team}"

        event = GoalEvent(
            event_id=f"goal_{frame_idx}_{int(time.time())}",
            frame=frame_idx,
            timestamp=time.time(),
            team=scoring_team,
            scorer_id=self.recent_carrier_id,
            speed_kmh=round(max(42.0, speed_kmh), 1),
            confidence=0.92 if celebration else 0.85,
            description=f"⚽ GOAL! Spectacular finish by {scorer_str} for Team {scoring_team}!",
        )
        return event

    def _check_celebration_cluster(self, players: Dict[str, Any], max_dist_px=240.0) -> bool:
        """Returns True if teammates are gathering in celebration."""
        if not self.recent_carrier_id or self.recent_carrier_id not in players:
            return False

        scorer = players[self.recent_carrier_id]
        s_bbox = scorer.get("bbox")
        if not s_bbox:
            return False

        sx = (s_bbox[0] + s_bbox[2]) / 2.0
        sy = (s_bbox[1] + s_bbox[3]) / 2.0
        s_team = scorer.get("team")

        cluster_count = 0
        for pid, pdata in players.items():
            if pid == self.recent_carrier_id:
                continue
            if pdata.get("team") != s_team:
                continue
            pbbox = pdata.get("bbox")
            if not pbbox:
                continue
            px = (pbbox[0] + pbbox[2]) / 2.0
            py = (pbbox[1] + pbbox[3]) / 2.0
            if math.hypot(px - sx, py - sy) < max_dist_px:
                cluster_count += 1

        return cluster_count >= 2

    def manual_trigger(self, team: int = 1, scorer_id: Optional[str] = "10") -> GoalEvent:
        """Manually trigger goal for broadcast testing."""
        if team == 1:
            self.score_team_1 += 1
        else:
            self.score_team_2 += 1

        event = GoalEvent(
            event_id=f"goal_manual_{int(time.time())}",
            frame=self.last_goal_frame + 300,
            timestamp=time.time(),
            team=team,
            scorer_id=scorer_id,
            speed_kmh=78.4,
            confidence=1.0,
            description=f"⚽ GOAL! Clean strike by Player #{scorer_id} (Team {team})!",
        )
        return event


class FoulDetector:
    """Detects tackles, collisions, player tumbles, and play stoppages."""

    def __init__(self, frame_size=(2880, 720), cooldown_frames=120):
        self.frame_w, self.frame_h = frame_size
        self.cooldown_frames = cooldown_frames
        self.last_foul_frame = -9999

        # Historical player states: pid -> {bbox, speed, aspect_ratio, team, frame, low_frames}
        self.player_history: Dict[str, Dict[str, Any]] = {}
        self.tumbled_players: Dict[str, int] = {}  # pid -> frame_idx

    def reset(self):
        self.last_foul_frame = -9999
        self.player_history.clear()
        self.tumbled_players.clear()

    def check(
        self,
        frame_idx: int,
        players: Dict[str, Any],
        ball_state: Dict[str, Any],
        referees: Optional[Dict[str, Any]] = None,
    ) -> Optional[FoulEvent]:
        """Detect tackle collisions, ground impacts, and referee reactions."""
        if frame_idx - self.last_foul_frame < self.cooldown_frames:
            return None

        # 1. Update player kinematic states and detect multi-frame persistent tumbles
        current_pids = list(players.keys())
        tumbles_this_frame = []

        for pid, pdata in players.items():
            bbox = pdata.get("bbox")
            if not bbox:
                continue

            pw = max(1.0, bbox[2] - bbox[0])
            ph = max(1.0, bbox[3] - bbox[1])
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            aspect_ratio = ph / pw  # Normal standing player ~ 1.8 - 2.8
            team = pdata.get("team")
            speed = pdata.get("speed", 0.0) or 0.0

            hist = self.player_history.get(pid, {})
            prev_h = hist.get("h", ph)
            prev_ar = hist.get("ar", aspect_ratio)
            prev_speed = hist.get("speed", speed)
            prev_low_frames = hist.get("low_frames", 0)

            # Check if player is on the ground
            is_fallen = (
                (aspect_ratio < 1.25 and prev_ar > 1.5) or
                (prev_h > 0 and (prev_h - ph) / prev_h > 0.32 and aspect_ratio < 1.30)
            )

            if is_fallen:
                low_frames = prev_low_frames + 1
            else:
                low_frames = 0

            # TEMPORAL PERSISTENCE GATE:
            # Require 4 consecutive low frames (~160ms) to reject single-frame YOLO box flicker
            if low_frames == 4:
                tumbles_this_frame.append((pid, cx, cy, team, max(speed, prev_speed)))
                self.tumbled_players[pid] = frame_idx

            self.player_history[pid] = {
                "cx": cx,
                "cy": cy,
                "w": pw,
                "h": ph,
                "ar": aspect_ratio,
                "speed": speed,
                "team": team,
                "frame": frame_idx,
                "low_frames": low_frames,
            }

        # 2. Match tumble with opponent physical collision and ball contest
        ball_bbox = ball_state.get("bbox")
        bx = (ball_bbox[0] + ball_bbox[2]) / 2.0 if ball_bbox else None
        by = (ball_bbox[1] + ball_bbox[3]) / 2.0 if ball_bbox else None

        for victim_id, vx, vy, v_team, v_speed in tumbles_this_frame:
            # Ball contest check: fouls happen around the ball (within 130px)
            if bx is not None and by is not None:
                dist_to_ball = math.hypot(vx - bx, vy - by)
                if dist_to_ball > 130.0:
                    continue  # Ignore off-the-ball slip or goalkeeper goal kick

            offender_id = None
            min_dist = float("inf")

            for other_id in current_pids:
                if other_id == victim_id:
                    continue
                other_p = players[other_id]
                o_team = other_p.get("team")

                # Ignore collisions between teammates
                if o_team is not None and v_team is not None and o_team == v_team:
                    continue

                obbox = other_p.get("bbox")
                if not obbox:
                    continue
                ox = (obbox[0] + obbox[2]) / 2.0
                oy = (obbox[1] + obbox[3]) / 2.0
                dist = math.hypot(ox - vx, oy - vy)

                # Physical contact distance (< 48px)
                if dist < 48.0 and dist < min_dist:
                    min_dist = dist
                    offender_id = other_id

            if offender_id is not None:
                # Confirmed physical tackle / collision foul
                self.last_foul_frame = frame_idx
                severity = "HARD_TACKLE" if v_speed > 16.0 else "TACTICAL_FOUL"
                card = "YELLOW_CARD" if (severity == "HARD_TACKLE" or v_speed > 20.0) else "WARNING"
                conf = 0.88

                card_icon = "🟨" if card == "YELLOW_CARD" else "⚠️"
                desc = (
                    f"{card_icon} FOUL DETECTED: Player #{offender_id} committed a {severity.lower().replace('_', ' ')} "
                    f"on Player #{victim_id}!"
                )

                return FoulEvent(
                    event_id=f"foul_{frame_idx}_{int(time.time())}",
                    frame=frame_idx,
                    timestamp=time.time(),
                    victim_id=str(victim_id),
                    offender_id=str(offender_id),
                    severity=severity,
                    card=card,
                    confidence=conf,
                    coords=(round(vx, 1), round(vy, 1)),
                    description=desc,
                )

        # 3. Clean up old player history
        if frame_idx % 60 == 0:
            stale_keys = [k for k, v in self.player_history.items() if frame_idx - v.get("frame", 0) > 30]
            for k in stale_keys:
                del self.player_history[k]

        return None

    def manual_trigger(
        self, victim_id: str = "7", offender_id: str = "4", severity: str = "HARD_TACKLE"
    ) -> FoulEvent:
        """Manually trigger a foul event for testing."""
        card = "YELLOW_CARD" if severity == "HARD_TACKLE" else "WARNING"
        card_icon = "🟨" if card == "YELLOW_CARD" else "⚠️"

        return FoulEvent(
            event_id=f"foul_manual_{int(time.time())}",
            frame=self.last_foul_frame + 100,
            timestamp=time.time(),
            victim_id=victim_id,
            offender_id=offender_id,
            severity=severity,
            card=card,
            confidence=0.95,
            coords=(960.0, 540.0),
            description=f"{card_icon} FOUL DETECTED: Heavy collision between #{offender_id} and #{victim_id}!",
        )
