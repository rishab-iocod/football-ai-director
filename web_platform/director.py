"""director.py — Multi-Camera Broadcast Director and Virtual PTZ Camera.

Features:
1. MultiCameraDirector: Autonomous scoring and hysteresis-gated switching across
   synchronized camera feeds (Cam 1: Left Wing, Cam 2: Center Midfield, Cam 3: Right Wing).
   Matches broadcast TV instincts: min shot length (2.0s), hysteresis (0.5s), ball tracking centrality.
2. VirtualPTZCamera: Cinematic smooth pan & zoom tracking with organic look-ahead,
   inertia damping, and dynamic ball/player focus modes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class CameraScore:
    cam_id: str
    cam_num: int
    score: float
    ball_status: str       # "detected", "predicted", "lost"
    ball_center: Optional[Tuple[float, float]]
    num_players: int
    is_live: bool


class MultiCameraDirector:
    """Evaluates multiple camera angles and makes professional TV switching decisions."""

    def __init__(
        self,
        num_cams: int = 3,
        fps: float = 25.0,
        min_shot_seconds: float = 1.0,
        hysteresis_seconds: float = 0.28,
        lost_fallback_seconds: float = 2.0,
        fallback_cam: int = 2,
        start_cam: int = 2,
        score_detected: float = 100.0,
        score_predicted: float = 60.0,
        score_center_max: float = 30.0,
        score_per_player: float = 1.0,
        score_players_max: float = 12.0,
    ):
        self.num_cams = num_cams
        self.fps = fps
        self.min_shot_seconds = min_shot_seconds
        self.hysteresis_seconds = hysteresis_seconds
        self.lost_fallback_seconds = lost_fallback_seconds
        self.fallback_cam = fallback_cam

        self.min_shot_frames = round(min_shot_seconds * fps)
        self.hysteresis_frames = max(1, round(hysteresis_seconds * fps))
        self.lost_frames_limit = round(lost_fallback_seconds * fps)

        # Scoring weights
        self.score_detected = score_detected
        self.score_predicted = score_predicted
        self.score_center_max = score_center_max
        self.score_per_player = score_per_player
        self.score_players_max = score_players_max

        # State machine
        self.current_cam = start_cam
        self.manual_override: Optional[int] = None  # None = Auto AI Director; 1, 2, 3 = forced
        self.shot_start_frame = 0
        self.challenger_cam: Optional[int] = None
        self.challenger_frames = 0
        self.lost_frames = 0
        self.total_cuts = 0
        self.ball_visible = False
        self.last_cut_reason = "Initial live coverage: Center Tactical (Cam 2)"
        self.live_reason = "Cam 2 live (Center Midfield coverage)"

        self.scores_history: List[CameraScore] = []

    def get_config(self) -> Dict[str, Any]:
        return {
            "min_shot_seconds": self.min_shot_seconds,
            "hysteresis_seconds": self.hysteresis_seconds,
            "lost_fallback_seconds": self.lost_fallback_seconds,
            "fallback_cam": self.fallback_cam,
            "score_detected": self.score_detected,
            "score_predicted": self.score_predicted,
            "score_center_max": self.score_center_max,
            "score_per_player": self.score_per_player,
            "score_players_max": self.score_players_max,
        }

    def update_config(self, cfg: Dict[str, Any]):
        if "min_shot_seconds" in cfg:
            self.min_shot_seconds = float(cfg["min_shot_seconds"])
            self.min_shot_frames = round(self.min_shot_seconds * self.fps)
        if "hysteresis_seconds" in cfg:
            self.hysteresis_seconds = float(cfg["hysteresis_seconds"])
            self.hysteresis_frames = max(1, round(self.hysteresis_seconds * self.fps))
        if "score_center_max" in cfg:
            self.score_center_max = float(cfg["score_center_max"])
        if "fallback_cam" in cfg:
            self.fallback_cam = int(cfg["fallback_cam"])

    def set_manual_override(self, cam_id: Optional[int]):
        """Manually force a camera (1, 2, 3) or pass None for Auto Director."""
        if cam_id in (1, 2, 3):
            if self.current_cam != cam_id:
                self.total_cuts += 1
                self.last_cut_reason = f"Manual Operator Cut to Cam {cam_id}"
                self.shot_start_frame = 0
            self.current_cam = cam_id
            self.manual_override = cam_id
            self.live_reason = f"Manual operator override on Cam {cam_id}"
        else:
            self.manual_override = None
            self.last_cut_reason = "Autonomous AI Director engaged"
            self.live_reason = "AI Director actively tracking ball & player formations"

    def score_camera(
        self,
        ball_status: str,
        ball_center: Optional[Tuple[float, float]],
        num_players: int,
        has_carrier: bool = False,
        cam_width: int = 960,
    ) -> float:
        """Score camera: ball carrier (+120), ball presence (+100/+60), centrality (+0..+30), player count (+2..+35)."""
        score = 0.0

        # Confirmed player with ball possession is the prime broadcast signal
        if has_carrier:
            score += 120.0

        if ball_status == "detected":
            score += self.score_detected
        elif ball_status == "predicted":
            score += self.score_predicted

        if ball_center is not None:
            # 1.0 when the ball is exactly in horizontal center, 0.0 at the edges
            offset = abs(ball_center[0] - cam_width / 2.0) / (cam_width / 2.0)
            score += self.score_center_max * max(0.0, 1.0 - offset)

        score += min(num_players * self.score_per_player, self.score_players_max)
        return round(score, 1)

    def update(
        self,
        frame_idx: int,
        cams_data: Dict[int, Dict[str, Any]],
        cam_width: int = 960,
    ) -> Tuple[int, List[CameraScore], bool, str]:
        """Returns (chosen_camera_id, scores_list, did_cut, live_decision_reason)."""
        scores_list: List[CameraScore] = []
        scores_by_cam: Dict[int, float] = {}
        balls_status: Dict[int, str] = {}

        for c_idx in range(1, self.num_cams + 1):
            c_info = cams_data.get(c_idx, {})
            b_st = c_info.get("ball", {})
            c_players = c_info.get("players", {})
            players_cnt = len(c_players)
            has_carrier = any(p.get("has_ball") for p in c_players.values())

            tracked = b_st.get("tracked", False)
            is_ext = b_st.get("is_extrapolated", False)
            b_src = "detected" if (tracked and not is_ext) else ("predicted" if (tracked and is_ext) else "lost")
            balls_status[c_idx] = b_src

            b_bbox = b_st.get("bbox")
            b_center = ((b_bbox[0] + b_bbox[2]) / 2.0, (b_bbox[1] + b_bbox[3]) / 2.0) if b_bbox else None

            sc = self.score_camera(b_src, b_center, players_cnt, has_carrier=has_carrier, cam_width=cam_width)
            scores_by_cam[c_idx] = sc

            scores_list.append(
                CameraScore(
                    cam_id=f"cam{c_idx}",
                    cam_num=c_idx,
                    score=sc,
                    ball_status=b_src,
                    ball_center=b_center,
                    num_players=players_cnt,
                    is_live=(c_idx == self.current_cam),
                )
            )

        self.scores_history = scores_list
        did_cut = False

        # If operator forced a manual camera cut, stay on it
        if self.manual_override is not None:
            self.live_reason = f"Manual override: Cam {self.manual_override} locked on-air"
            return self.manual_override, scores_list, False, self.live_reason

        # Autonomous AI Director Evaluation
        best_cam = max(scores_by_cam, key=lambda k: scores_by_cam[k])
        has_any_ball = any(st in ("detected", "predicted") for st in balls_status.values())

        shot_duration_frames = frame_idx - self.shot_start_frame
        shot_duration_sec = shot_duration_frames / self.fps
        can_cut = shot_duration_frames >= self.min_shot_frames

        if has_any_ball:
            self.lost_frames = 0
            if best_cam == self.current_cam:
                self.challenger_cam = None
                self.challenger_frames = 0
                self.live_reason = (
                    f"Cam {self.current_cam} has best score ({scores_by_cam[self.current_cam]:.1f}) | "
                    f"Shot time: {shot_duration_sec:.1f}s"
                )
            else:
                # Challenger camera is beating the live camera
                if best_cam == self.challenger_cam:
                    self.challenger_frames += 1
                else:
                    self.challenger_cam = best_cam
                    self.challenger_frames = 1

                hyst_sec = self.challenger_frames / self.fps
                if self.challenger_frames < self.hysteresis_frames:
                    self.live_reason = (
                        f"Cam {best_cam} leads ({scores_by_cam[best_cam]:.1f} vs {scores_by_cam[self.current_cam]:.1f}) "
                        f"— Hysteresis: {hyst_sec:.1f}s/{self.hysteresis_seconds}s"
                    )
                elif not can_cut:
                    remaining_lock = max(0.0, self.min_shot_seconds - shot_duration_sec)
                    self.live_reason = (
                        f"Cam {best_cam} leads — Min shot length locked ({remaining_lock:.1f}s remaining)"
                    )
                else:
                    # EXECUTE BROADCAST CUT!
                    old_cam = self.current_cam
                    self.current_cam = best_cam
                    self.shot_start_frame = frame_idx
                    self.challenger_cam = None
                    self.challenger_frames = 0
                    self.total_cuts += 1
                    self.last_cut_reason = f"CUT: Cam {best_cam} scored higher for {self.hysteresis_seconds}s"
                    self.live_reason = f"CUT to Cam {best_cam}: Scored {scores_by_cam[best_cam]:.1f} vs Cam {old_cam}"
                    did_cut = True
        else:
            self.lost_frames += 1
            lost_sec = self.lost_frames / self.fps
            if self.lost_frames > self.lost_frames_limit and self.current_cam != self.fallback_cam:
                if can_cut:
                    old_cam = self.current_cam
                    self.current_cam = self.fallback_cam
                    self.shot_start_frame = frame_idx
                    self.challenger_cam = None
                    self.challenger_frames = 0
                    self.total_cuts += 1
                    self.last_cut_reason = f"Fallback to tactical center Cam {self.fallback_cam} (ball lost)"
                    self.live_reason = f"Fallback CUT to Cam {self.fallback_cam}: Ball lost on all cameras ({lost_sec:.1f}s)"
                    did_cut = True
                else:
                    self.live_reason = f"Ball lost ({lost_sec:.1f}s) — Waiting min shot length"
            else:
                self.live_reason = f"Ball searching across feeds ({lost_sec:.1f}s) | Cam {self.current_cam} live"

        return self.current_cam, scores_list, did_cut, self.live_reason


class VirtualPTZCamera:
    """Dynamic Pan & Zoom engine that tracks balls and players with broadcast cinematography."""

    MODE_AUTO = "DIRECTOR_AUTO"
    MODE_BALL = "BALL_FOCUS"
    MODE_PLAYER = "PLAYER_FOCUS"
    MODE_MANUAL = "MANUAL_PTZ"

    def __init__(
        self,
        frame_size: Tuple[int, int] = (1920, 1080),
        output_size: Tuple[int, int] = (1280, 720),
        smooth_alpha: float = 0.15,
        zoom: float = 1.0,
    ):
        self.frame_w, self.frame_h = frame_size
        self.out_w, self.out_h = output_size
        self.smooth_alpha = smooth_alpha
        self.smooth_alpha_x = smooth_alpha
        self.smooth_alpha_y = 0.04
        self.deadband_y = 18.0
        self.zoom_hold_timer = 0

        # Camera viewport center & zoom
        self.cam_x = self.frame_w / 2.0
        self.cam_y = self.frame_h * 0.40  # Natural television horizon
        self.current_zoom = zoom
        self.target_zoom = zoom

        # Manual pan offsets
        self.manual_pan_x = 0.0
        self.manual_pan_y = 0.0

        # Mode
        self.mode = self.MODE_AUTO
        self.target_player_id: Optional[str] = None
        self.hero_info: Dict[str, Any] = {}

        # Look-ahead velocity tracking
        self.prev_target_x = self.cam_x
        self.prev_target_y = self.cam_y
        self.target_vx = 0.0
        self.target_vy = 0.0

        # Goal celebration timer
        self.celebration_frames = 0

    def set_mode(self, mode: str, player_id: Optional[str] = None):
        """Set camera tracking mode."""
        if mode in (self.MODE_AUTO, self.MODE_BALL, self.MODE_PLAYER, self.MODE_MANUAL):
            self.mode = mode
            if player_id is not None:
                self.target_player_id = str(player_id)

    def set_zoom(self, zoom: float):
        """Set target zoom scale (1.0 = wide, up to 3.2 = close-up)."""
        self.target_zoom = float(np.clip(zoom, 1.0, 3.2))

    def set_manual_pan(self, pan_x: float, pan_y: float):
        """Manual pan offsets in normalized space [-1.0, 1.0]."""
        self.manual_pan_x = float(np.clip(pan_x, -1.0, 1.0)) * (self.frame_w / 2.0)
        self.manual_pan_y = float(np.clip(pan_y, -1.0, 1.0)) * (self.frame_h / 2.0)

    def trigger_celebration(self, player_id: Optional[str] = None, duration_frames: int = 150):
        """Spotlight goalscorer with celebration zoom."""
        self.celebration_frames = duration_frames
        self.target_player_id = str(player_id) if player_id else None
        self.hero_info = {"title": "GOAL CELEBRATION", "player_id": self.target_player_id}

    def update(
        self,
        frame_idx: int,
        ball_state: Dict[str, Any],
        players: Dict[str, Any],
    ) -> Tuple[Tuple[int, int, int, int], Dict[str, Any]]:
        """Calculate the smoothed (x1, y1, x2, y2) crop box and director metadata."""
        if self.celebration_frames > 0:
            self.celebration_frames -= 1

        tx, ty, tz = self._determine_target(ball_state, players)

        dt_vx = tx - self.prev_target_x
        self.target_vx = 0.8 * self.target_vx + 0.2 * dt_vx
        self.prev_target_x = tx

        # Horizontal lookahead only (never lead vertically)
        lead_x = tx + np.clip(self.target_vx * 3.0, -120.0, 120.0)

        # Smooth horizontal pan
        self.cam_x += self.smooth_alpha_x * (lead_x - self.cam_x)

        # Stable vertical horizon (no lead overshoot, deadband threshold, gentle damping)
        if abs(ty - self.cam_y) > self.deadband_y:
            self.cam_y += self.smooth_alpha_y * (ty - self.cam_y)

        # Smooth stable zoom with deadband to completely eliminate zoom breathing/pulsing
        zoom_diff = tz - self.current_zoom
        if abs(zoom_diff) > 0.08:
            if zoom_diff > 0:
                self.zoom_hold_timer = 40  # hold framing on action for 1.6s
                self.current_zoom += 0.03 * zoom_diff
            elif self.zoom_hold_timer > 0:
                self.zoom_hold_timer -= 1
            else:
                self.current_zoom += 0.015 * zoom_diff

        crop_box = self._compute_crop_box()

        meta = {
            "mode": self.mode,
            "zoom": round(self.current_zoom, 2),
            "cam_center": (round(self.cam_x, 1), round(self.cam_y, 1)),
            "hero_player_id": self.target_player_id,
            "is_celebrating": self.celebration_frames > 0,
        }
        return crop_box, meta

    def _determine_target(
        self, ball_state: Dict[str, Any], players: Dict[str, Any]
    ) -> Tuple[float, float, float]:
        ball_bbox = ball_state.get("bbox")
        bx = (ball_bbox[0] + ball_bbox[2]) / 2.0 if ball_bbox else None
        by = (ball_bbox[1] + ball_bbox[3]) / 2.0 if ball_bbox else None

        if self.celebration_frames > 0:
            if self.target_player_id and self.target_player_id in players:
                pbox = players[self.target_player_id].get("bbox")
                if pbox:
                    px = (pbox[0] + pbox[2]) / 2.0
                    py = (pbox[1] + pbox[3]) / 2.0
                    return px, py, 2.2
            if bx and by:
                return bx, by, 1.8

        if self.mode == self.MODE_BALL:
            if bx is not None and by is not None:
                return bx, by, max(1.8, self.target_zoom)
            return self.cam_x, self.cam_y, max(1.6, self.target_zoom)

        if self.mode == self.MODE_PLAYER:
            target_p = None
            if self.target_player_id and self.target_player_id in players:
                target_p = players[self.target_player_id]
            else:
                for pid, pdata in players.items():
                    if pdata.get("has_ball"):
                        target_p = pdata
                        self.target_player_id = pid
                        break
                if not target_p and players:
                    target_p = max(players.values(), key=lambda p: p.get("speed", 0.0) or 0.0)

            if target_p and target_p.get("bbox"):
                pbox = target_p["bbox"]
                px = (pbox[0] + pbox[2]) / 2.0
                py = (pbox[1] + pbox[3]) / 2.0
                return px, py, max(1.6, self.target_zoom)

            if bx and by:
                return bx, by, self.target_zoom
            return self.frame_w / 2.0, self.frame_h / 2.0, 1.0

        if self.mode == self.MODE_MANUAL:
            mx = self.frame_w / 2.0 + self.manual_pan_x
            my = self.frame_h / 2.0 + self.manual_pan_y
            return mx, my, self.target_zoom

        # Auto Director: dynamically pan and zoom to player who has the ball
        ball_carrier = None
        if players:
            for p in players.values():
                if p.get("has_ball") and p.get("bbox"):
                    ball_carrier = p
                    break

        if ball_carrier is not None:
            cb = ball_carrier["bbox"]
            cx = (cb[0] + cb[2]) / 2.0
            cy = float(np.clip((cb[1] + cb[3]) / 2.0, self.frame_h * 0.28, self.frame_h * 0.52))
            # Cinematic television framing on the ball carrier
            return cx, cy, max(1.28, self.target_zoom)

        if bx is not None and by is not None:
            cy = float(np.clip(by, self.frame_h * 0.28, self.frame_h * 0.52))
            return bx, cy, max(1.20, self.target_zoom)

        if players:
            xs = [(p["bbox"][0] + p["bbox"][2]) / 2.0 for p in players.values() if p.get("bbox")]
            ys = [(p["bbox"][1] + p["bbox"][3]) / 2.0 for p in players.values() if p.get("bbox")]
            if xs and ys:
                cy = float(np.clip(np.median(ys), self.frame_h * 0.28, self.frame_h * 0.52))
                return float(np.median(xs)), cy, max(1.15, self.target_zoom)

        return self.cam_x, self.cam_y, self.current_zoom

    def _compute_crop_box(self) -> Tuple[int, int, int, int]:
        target_aspect = 16.0 / 9.0
        frame_aspect = self.frame_w / max(1.0, self.frame_h)

        if frame_aspect >= target_aspect:
            # Panoramic / wide source: crop a true 16:9 broadcast window
            crop_h = self.frame_h / max(1.0, self.current_zoom)
            crop_w = crop_h * target_aspect
        else:
            # 4:3 camera source (e.g. 960x720): crop a true 16:9 window zoomed on ball/player
            crop_w = self.frame_w / max(1.0, self.current_zoom)
            crop_h = crop_w / target_aspect
            if crop_h > self.frame_h:
                crop_h = self.frame_h
                crop_w = crop_h * target_aspect

        x1 = self.cam_x - crop_w / 2.0
        y1 = self.cam_y - crop_h / 2.0

        if x1 < 0:
            x1 = 0
        elif x1 + crop_w > self.frame_w:
            x1 = self.frame_w - crop_w

        if y1 < 0:
            y1 = 0
        elif y1 + crop_h > self.frame_h:
            y1 = self.frame_h - crop_h

        return int(x1), int(y1), int(x1 + crop_w), int(y1 + crop_h)

    def render_ptz_frame(
        self,
        frame: np.ndarray,
        crop_box: Tuple[int, int, int, int],
        meta: Dict[str, Any],
        title: Optional[str] = None,
    ) -> np.ndarray:
        x1, y1, x2, y2 = crop_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

        cropped = frame[y1:y2, x1:x2]
        if cropped.size == 0:
            return cv2.resize(frame, (self.out_w, self.out_h))

        return cv2.resize(cropped, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
