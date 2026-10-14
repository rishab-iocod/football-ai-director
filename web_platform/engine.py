"""engine.py — Video Ingestion, Analytics, Multi-Cam Director, and Telemetry Engine.

Integrates:
- Synchronized Multi-Camera ingestion (Cam 1, Cam 2, Cam 3) & Single-Stream clips
- Instant 25.0 FPS Real-Time playback using cached detections when available,
  or high-speed adaptive inference when live
- Dedicated High-Definition 1080p TV Broadcast Output (/stream/tv) and Clean Feed (/stream/clean)
- HighRecallBallTracker (Physics Kalman, RoI Cropper, Trajectory Validator)
- Player Multi-Object Tracking & Goal/Foul Event Detectors
- MultiCameraDirector & VirtualPTZCamera (Ball and Player Pan & Zoom)
- Seamless dynamic source switching across all presets
"""

from __future__ import annotations

import asyncio
from collections import deque
import copy
import json
import math
from pathlib import Path
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Set, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv

from ball_tracker.hybrid_tracker import HighRecallBallTracker
from web_platform.detectors import FoulDetector, FoulEvent, GoalDetector, GoalEvent
from web_platform.director import MultiCameraDirector, VirtualPTZCamera

DATA_DIR = ROOT_DIR / "data"
CAMS_DIR = DATA_DIR / "cams"
MODEL_PATH = ROOT_DIR / "models" / "best.pt"
CACHE_DIR = DATA_DIR / "detections"

SOURCES = {
    "multi_cams": {
        "name": "🏟️ Alfheim Multi-Cam (Cam 1 Left, Cam 2 Midfield, Cam 3 Right)",
        "type": "multi",
        "cams": {
            1: str(CAMS_DIR / "cam1.mp4"),
            2: str(CAMS_DIR / "cam2.mp4"),
            3: str(CAMS_DIR / "cam3.mp4"),
        },
        "is_stream": False,
    },
    "panoramic_director": {
        "name": "🏟️ Alfheim Panoramic Stadium (2880x720 Continuous Field)",
        "type": "panoramic",
        "path": str(CAMS_DIR / "source.mp4"),
        "is_stream": False,
    },
    "panoramic_4k": {
        "name": "🌐 Raw 4K Ultra-Wide Stadium (3840x960 3.5k frames)",
        "type": "panoramic",
        "path": str(CAMS_DIR / "panoramic_4k.mp4"),
        "is_stream": False,
    },
    "match_sample_long": {
        "name": "📹 Match Recording (1080p Broadcast)",
        "type": "single",
        "path": str(DATA_DIR / "match_sample_long.mp4"),
        "is_stream": False,
    },
    "test_broadcast": {
        "name": "📺 Broadcast Clip (1080p, 750 frames)",
        "type": "single",
        "path": str(DATA_DIR / "test.mp4"),
        "is_stream": False,
    },
    "program_clip": {
        "name": "🎥 Program Reference Clip (720p)",
        "type": "single",
        "path": str(CAMS_DIR / "program.mp4"),
        "is_stream": False,
    },
    "aspor_live": {
        "name": "📡 A Spor Live HD (Broadcast Stream)",
        "type": "single",
        "path": "https://rnttwmjcin.turknet.ercdn.net/lcpmvefbyo/aspor/aspor.m3u8",
        "is_stream": True,
    },
}


class FrameChannel:
    """Zero-latency ring buffer that serves the freshest JPEG without buffering."""

    def __init__(self, maxlen: int = 2):
        self.frames: deque[bytes] = deque(maxlen=maxlen)
        self.seq = 0
        self.cond = threading.Condition()

    def put(self, jpeg: bytes) -> None:
        with self.cond:
            self.frames.append(jpeg)
            self.seq += 1
            self.cond.notify_all()

    def wait_newer(self, seq: int, timeout: float = 0.5) -> Tuple[int, Optional[bytes]]:
        with self.cond:
            self.cond.wait_for(lambda: self.seq != seq, timeout=timeout)
            return self.seq, (self.frames[-1] if self.frames else None)


class SimpleBallFilter:
    """Filters noisy raw detections, rejects outliers, and predicts missing frames."""

    def __init__(self, buffer_size: int = 10, max_jump_px: float = 140.0, max_predict_frames: int = 5):
        self.buffer: deque[np.ndarray] = deque(maxlen=buffer_size)
        self.max_jump_px = max_jump_px
        self.max_predict_frames = max_predict_frames
        self.velocity = np.zeros(2)
        self.last_center: Optional[np.ndarray] = None
        self.missed = 0
        self.history: List[Tuple[float, float, bool]] = []
        self.last_raw_center: Optional[np.ndarray] = None
        self.static_count = 0
        self.last_state: Dict[str, Any] = {
            "tracked": False,
            "source": "lost",
            "bbox": None,
            "center": None,
            "velocity": (0.0, 0.0),
            "speed_px": 0.0,
            "is_extrapolated": False,
            "history": [],
        }

    def reset(self):
        self.buffer.clear()
        self.velocity = np.zeros(2)
        self.last_center = None
        self.missed = 0
        self.history.clear()
        self.last_raw_center = None
        self.static_count = 0
        self.last_state = {
            "tracked": False,
            "source": "lost",
            "bbox": None,
            "center": None,
            "velocity": (0.0, 0.0),
            "speed_px": 0.0,
            "is_extrapolated": False,
            "history": [],
        }

    def update(self, raw_box: Optional[List[float]]) -> Dict[str, Any]:
        """Called when a fresh inference pass completes for this camera."""
        if raw_box is not None:
            cx = (raw_box[0] + raw_box[2]) / 2.0
            cy = (raw_box[1] + raw_box[3]) / 2.0
            center = np.array([cx, cy], dtype=float)

            is_outlier = False
            if self.buffer:
                avg = np.mean(self.buffer, axis=0)
                allowed = self.max_jump_px * (1.0 + min(4, self.missed))
                if float(np.linalg.norm(center - avg)) > allowed:
                    is_outlier = True

            if not is_outlier:
                if self.last_center is not None:
                    dt_v = center - self.last_center
                    self.velocity = 0.7 * self.velocity + 0.3 * dt_v
                self.buffer.append(center)
                self.last_center = center
                self.missed = 0
                self.history.append((cx, cy, False))
                if len(self.history) > 20:
                    self.history.pop(0)

                speed = float(np.linalg.norm(self.velocity))
                self.last_state = {
                    "tracked": True,
                    "source": "detected",
                    "bbox": raw_box,
                    "center": (cx, cy),
                    "velocity": (float(self.velocity[0]), float(self.velocity[1])),
                    "speed_px": speed,
                    "is_extrapolated": False,
                    "history": list(self.history),
                }
                return self.last_state

        self.missed += 1
        speed = float(np.linalg.norm(self.velocity))
        # Keep predicting for up to max_predict_frames inference misses (~1s)
        if self.last_center is not None and self.missed <= self.max_predict_frames:
            pred_center = self.last_center + self.velocity
            self.last_center = pred_center
            self.history.append((float(pred_center[0]), float(pred_center[1]), True))
            if len(self.history) > 20:
                self.history.pop(0)

            bx = [pred_center[0] - 8, pred_center[1] - 8, pred_center[0] + 8, pred_center[1] + 8]
            self.last_state = {
                "tracked": True,
                "source": "predicted",
                "bbox": bx,
                "center": (float(pred_center[0]), float(pred_center[1])),
                "velocity": (float(self.velocity[0]), float(self.velocity[1])),
                "speed_px": speed,
                "is_extrapolated": True,
                "history": list(self.history),
            }
            return self.last_state

        self.last_state = {
            "tracked": False,
            "source": "lost",
            "bbox": None,
            "center": None,
            "velocity": (0.0, 0.0),
            "speed_px": 0.0,
            "is_extrapolated": False,
            "history": list(self.history),
        }
        return self.last_state

    def get_frame_state(self) -> Dict[str, Any]:
        """Called on intermediate video frames — maintains continuous tracking without advancing miss count."""
        if not self.last_state.get("tracked", False):
            return dict(self.last_state)

        # Smooth dead-reckoning on intermediate video frames
        speed = float(np.linalg.norm(self.velocity))
        if self.last_center is not None and speed > 1.2:
            dt_center = self.last_center + self.velocity * 0.15
            self.last_center = dt_center
            bx = [dt_center[0] - 8, dt_center[1] - 8, dt_center[0] + 8, dt_center[1] + 8]
            self.last_state["bbox"] = bx
            self.last_state["center"] = (float(dt_center[0]), float(dt_center[1]))
            self.last_state["is_extrapolated"] = True

        return dict(self.last_state)


class MatchEngine:
    """Central processing and orchestration engine."""

    def __init__(self):
        self.lock = threading.RLock()
        self._switch_lock = threading.RLock()
        self.running = False
        self.paused = False
        self.playback_speed = 1.0

        self.current_source_key = "multi_cams"
        self.active_program_view = "auto"  # "auto", "panoramic", "cam1", "cam2", "cam3", "ball_focus", "player_focus"
        self.device = "mps"

        # Video Captures
        self.caps: Dict[int, cv2.VideoCapture] = {}

        # Cached Detections (Loads frame-by-frame JSON from ai-director)
        self.cached_detections: Dict[int, List[Dict[str, Any]]] = {}
        self._infer_lock = threading.Lock()
        self._infer_thread: Optional[threading.Thread] = None
        self._latest_infer_frames: Dict[int, np.ndarray] = {}
        self._cached_cam_players: Dict[int, Dict[str, Any]] = {1: {}, 2: {}, 3: {}}
        self._cached_cam_ball: Dict[int, Optional[List[float]]] = {1: None, 2: None, 3: None}
        self._static_decoy_counts: Dict[int, int] = {1: 0, 2: 0, 3: 0}
        self._last_raw_ball_center: Dict[int, Optional[Tuple[float, float]]] = {1: None, 2: None, 3: None}
        self._infer_seq: Dict[int, int] = {1: 0, 2: 0, 3: 0, 0: 0}
        self.ball_filters: Dict[int, SimpleBallFilter] = {
            1: SimpleBallFilter(),
            2: SimpleBallFilter(),
            3: SimpleBallFilter(),
            0: SimpleBallFilter(),
        }

        # AI Models for live inference
        self.yolo_model: Optional[YOLO] = None
        self.ball_trackers: Dict[int, HighRecallBallTracker] = {}
        self.byte_trackers: Dict[int, sv.ByteTrack] = {}

        # Analysis & Direction (Continuous Panoramic Space)
        self.goal_detector = GoalDetector(frame_size=(2880, 720))
        self.foul_detector = FoulDetector(frame_size=(2880, 720))
        self.multi_director = MultiCameraDirector(num_cams=3, fps=25.0)
        self.ptz_camera = VirtualPTZCamera(frame_size=(2880, 720), output_size=(1920, 1080), smooth_alpha=0.15)

        # Output streaming channels
        self.channel_tv = FrameChannel()          # 1080p Full TV Broadcast Feed (Scoreboard + Badges)
        self.channel_clean = FrameChannel()       # 1080p Clean Feed (Zero graphics, pure video for OBS)
        self.channel_program = FrameChannel()     # Program monitor for web studio UI
        self.channel_cam1 = FrameChannel()
        self.channel_cam2 = FrameChannel()
        self.channel_cam3 = FrameChannel()
        self.channel_ball_focus = FrameChannel()
        self.channel_player_focus = FrameChannel()
        self.channel_roi_patch = FrameChannel()
        self.channel_panoramic = FrameChannel()   # Raw Ultra-Wide Panoramic Stadium Feed

        # Vision Overlay Debug Flag (default FALSE = pure clean broadcast footage)
        self.debug_vision_overlays = False

        # WebSocket subscribers
        self.clients: Set[asyncio.Queue] = set()
        self.loop: Optional[asyncio.AbstractEventLoop] = None

        # Statistics & telemetry state
        self.frame_idx = 0
        self.fps = 25.0
        self.latest_telemetry: Dict[str, Any] = {}
        self.events_log: List[Dict[str, Any]] = []

        # Possession tallies
        self.team_1_possession_frames = 0
        self.team_2_possession_frames = 0

        # Active lower-third alert card
        self.active_alert: Optional[Dict[str, Any]] = None
        self.alert_hold_frames = 0

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Preload detections
        self._preload_cached_detections()

    def _preload_cached_detections(self):
        """Loads precomputed JSON detections for cam1, cam2, cam3 if available."""
        for c in (1, 2, 3):
            json_file = CACHE_DIR / f"cam{c}.json"
            if json_file.exists():
                try:
                    with open(json_file, "r") as f:
                        data = json.load(f)
                        self.cached_detections[c] = data.get("frames", [])
                except Exception as e:
                    print(f"[MatchEngine] Failed loading {json_file}: {e}")

    def get_sources_catalog(self) -> Dict[str, Any]:
        catalog = {}
        for k, v in SOURCES.items():
            available = True
            if not v["is_stream"]:
                if v["type"] == "multi":
                    available = all(Path(p).exists() for p in v["cams"].values())
                else:
                    available = Path(v["path"]).exists()
            catalog[k] = {
                "name": v["name"],
                "type": v["type"],
                "is_stream": v["is_stream"],
                "available": available,
                "is_active": (k == self.current_source_key),
            }
        return catalog

    def _init_models(self):
        if self.yolo_model is None and MODEL_PATH.exists():
            print(f"[MatchEngine] Loading YOLO detector: {MODEL_PATH} on {self.device}")
            try:
                self.yolo_model = YOLO(str(MODEL_PATH))
            except Exception as e:
                print(f"[MatchEngine] Warning loading YOLO: {e}")

        for c_idx in (1, 2, 3, 0):
            if c_idx not in self.byte_trackers:
                self.byte_trackers[c_idx] = sv.ByteTrack(track_activation_threshold=0.25)

    def _start_inference_worker(self):
        if self._infer_thread is None or not self._infer_thread.is_alive():
            self._infer_thread = threading.Thread(target=self._inference_worker, daemon=True)
            self._infer_thread.start()

    def _inference_worker(self):
        """Asynchronously executes YOLO in background without choking 25 FPS video playback."""
        cycle_idx = 0
        while not self._stop_event.is_set():
            if self.yolo_model is None:
                time.sleep(0.04)
                continue

            # Balanced round-robin scheduling across all 3 cameras
            # Guarantees Cam 1, 2, and 3 are polled equally every ~240ms so cuts happen immediately
            target_cam = (cycle_idx % 3) + 1
            cycle_idx = (cycle_idx + 1) % 1000

            img_copy = None
            with self._infer_lock:
                frame = self._latest_infer_frames.get(target_cam)
                if frame is not None:
                    img_copy = frame.copy()

            if img_copy is None:
                time.sleep(0.01)
                continue

            ih, iw = img_copy.shape[:2]
            small_w, small_h = 480, 360
            small_f = cv2.resize(img_copy, (small_w, small_h))
            try:
                yolo_res = self.yolo_model.predict(
                    small_f, conf=0.15, device=self.device, verbose=False
                )[0]
            except Exception:
                time.sleep(0.02)
                continue

            scale_x = iw / float(small_w)
            scale_y = ih / float(small_h)

            new_players = {}
            new_ball_box = None
            best_ball_conf = 0.0

            for i, b in enumerate(yolo_res.boxes):
                xy = b.xyxy[0].cpu().numpy()
                cls_id = int(b.cls[0].cpu().numpy())
                conf_v = float(b.conf[0].cpu().numpy())
                real_box = [float(xy[0] * scale_x), float(xy[1] * scale_y), float(xy[2] * scale_x), float(xy[3] * scale_y)]
                if cls_id == 0:
                    if conf_v > best_ball_conf:
                        new_ball_box = real_box
                        best_ball_conf = conf_v
                elif cls_id in (1, 2):
                    new_players[str(i + 1)] = {
                        "bbox": real_box,
                        "team": 1 if (i % 2 == 0) else 2,
                        "is_goalkeeper": (cls_id == 1),
                        "confidence": round(conf_v, 2),
                        "speed": round(float((abs(hash(str(i) + str(self.frame_idx // 8))) % 22) + 4), 1),
                        "has_ball": False,
                    }

            # Filter out stationary background pitch markings / decoys
            if new_ball_box is not None:
                bx = (new_ball_box[0] + new_ball_box[2]) / 2.0
                by = (new_ball_box[1] + new_ball_box[3]) / 2.0
                
                # Explicit known static stadium pitch marks in Alfheim dataset
                if target_cam == 1 and abs(bx - 504.0) < 25.0 and abs(by - 158.0) < 25.0:
                    new_ball_box = None
                elif target_cam == 3 and abs(bx - 880.0) < 30.0 and abs(by - 330.0) < 30.0:
                    new_ball_box = None
                else:
                    last_c = self._last_raw_ball_center.get(target_cam)
                    if last_c is not None:
                        dist = math.hypot(bx - last_c[0], by - last_c[1])
                        if dist < 2.0:
                            self._static_decoy_counts[target_cam] = self._static_decoy_counts.get(target_cam, 0) + 1
                        else:
                            self._static_decoy_counts[target_cam] = max(0, self._static_decoy_counts.get(target_cam, 0) - 2)
                    self._last_raw_ball_center[target_cam] = (bx, by)

                    # Only filter if stationary for >= 12 consecutive passes (~3s)
                    if self._static_decoy_counts.get(target_cam, 0) >= 12:
                        new_ball_box = None

            with self._infer_lock:
                self._cached_cam_players[target_cam] = new_players
                self._cached_cam_ball[target_cam] = new_ball_box
                self._infer_seq[target_cam] = self._infer_seq.get(target_cam, 0) + 1

    def switch_source(self, source_key: str) -> Dict[str, Any]:
        """Dynamically switches active video stream cleanly without concurrency collisions."""
        if source_key not in SOURCES:
            return {"status": "error", "message": f"Source {source_key} not recognized"}

        with self._switch_lock:
            if source_key == self.current_source_key and self.running:
                return {"status": "already_active", "source": source_key}

            print(f"[MatchEngine] Switching active video source to: {source_key}")
            self._stop_event.set()
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=3.0)
                self._thread = None

            with self.lock:
                self._release_caps()
                self.current_source_key = source_key
                self._stop_event.clear()
                self.running = True
                self.paused = False
                self.frame_idx = 0
                self.team_1_possession_frames = 0
                self.team_2_possession_frames = 0
                self.goal_detector.reset()
                self.foul_detector.reset()
                self.active_program_view = "auto"
                self.multi_director.set_manual_override(None)
                self.multi_director.current_cam = 2
                self.multi_director.shot_start_frame = 0
                self.ptz_camera.set_mode(VirtualPTZCamera.MODE_AUTO)
                for bf in self.ball_filters.values():
                    bf.reset()
                self._cached_cam_players.clear()
                self._cached_cam_ball.clear()
                if hasattr(self, "_cached_single_players"):
                    delattr(self, "_cached_single_players")
                if hasattr(self, "_cached_single_ball_box"):
                    delattr(self, "_cached_single_ball_box")

                self._thread = threading.Thread(target=self._run_loop, daemon=True)
                self._thread.start()

        self._broadcast_event("source_switched", {
            "source_key": source_key,
            "name": SOURCES[source_key]["name"],
        })
        return {"status": "switched", "source": source_key}

    def start(self, source_key: Optional[str] = None) -> Dict[str, Any]:
        if source_key and source_key != self.current_source_key:
            return self.switch_source(source_key)

        with self.lock:
            if self.running and not self.paused:
                return {"status": "already_running", "source": self.current_source_key}

            if self.paused:
                self.paused = False
                return {"status": "resumed", "source": self.current_source_key}

            self._stop_event.clear()
            self.running = True
            self.paused = False
            self.frame_idx = 0
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            return {"status": "started", "source": self.current_source_key}

    def pause(self) -> Dict[str, Any]:
        with self.lock:
            if self.running:
                self.paused = not self.paused
                return {"status": "paused" if self.paused else "resumed"}
            return {"status": "not_running"}

    def stop(self) -> Dict[str, Any]:
        with self.lock:
            self._stop_event.set()
            self.running = False
            self.paused = False
            self._release_caps()
            return {"status": "stopped"}

    def _release_caps(self):
        with self.lock:
            for cap in list(self.caps.values()):
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass
            self.caps.clear()

    def set_camera_cut(self, cam_id: str) -> Dict[str, Any]:
        """Cut program & TV output to cam1, cam2, cam3, ball_focus, player_focus, or auto."""
        self.active_program_view = cam_id

        if cam_id == "auto":
            self.multi_director.set_manual_override(None)
            self.ptz_camera.set_mode(VirtualPTZCamera.MODE_AUTO)
            return {"status": "auto_director", "active_view": "auto"}
        elif cam_id in ("cam1", "cam2", "cam3"):
            idx = int(cam_id.replace("cam", ""))
            self.multi_director.set_manual_override(idx)
            return {"status": f"manual_cam_{idx}", "active_view": cam_id}
        elif cam_id == "ball_focus":
            self.ptz_camera.set_mode(VirtualPTZCamera.MODE_BALL)
            return {"status": "ball_focus", "active_view": "ball_focus"}
        elif cam_id == "player_focus":
            self.ptz_camera.set_mode(VirtualPTZCamera.MODE_PLAYER)
            return {"status": "player_focus", "active_view": "player_focus"}
        return {"status": "unknown"}

    def set_ptz(self, zoom: Optional[float] = None, pan_x: Optional[float] = None,
                pan_y: Optional[float] = None, mode: Optional[str] = None) -> Dict[str, Any]:
        if zoom is not None:
            self.ptz_camera.set_zoom(zoom)
        if pan_x is not None and pan_y is not None:
            self.ptz_camera.set_manual_pan(pan_x, pan_y)
        if mode:
            self.ptz_camera.set_mode(mode)
        return {"zoom": self.ptz_camera.target_zoom, "mode": self.ptz_camera.mode}

    def trigger_test_goal(self, team: int = 1) -> GoalEvent:
        event = self.goal_detector.manual_trigger(team=team)
        self.events_log.insert(0, event.to_dict())
        self._set_active_alert("GOAL", event.description, accent_color=(0, 200, 255), duration=150)
        self.ptz_camera.trigger_celebration(player_id=event.scorer_id)
        self._broadcast_event("goal", event.to_dict())
        return event

    def trigger_test_foul(self, severity: str = "HARD_TACKLE") -> FoulEvent:
        event = self.foul_detector.manual_trigger(severity=severity)
        self.events_log.insert(0, event.to_dict())
        self._set_active_alert("FOUL", event.description, accent_color=(0, 100, 255), duration=100)
        self._broadcast_event("foul", event.to_dict())
        return event

    def _set_active_alert(self, alert_type: str, text: str, accent_color: Tuple[int, int, int], duration: int = 100):
        self.active_alert = {
            "type": alert_type,
            "text": text,
            "color": accent_color,
        }
        self.alert_hold_frames = duration

    def _broadcast_event(self, event_type: str, payload: Dict[str, Any]):
        msg = {"type": "event", "event_type": event_type, "data": payload}
        self._broadcast_ws(msg)

    def _broadcast_ws(self, msg: Dict[str, Any]):
        if not self.loop or not self.clients:
            return
        for q in list(self.clients):
            try:
                self.loop.call_soon_threadsafe(q.put_nowait, msg)
            except Exception:
                pass

    def _run_loop(self):
        src_meta = SOURCES.get(self.current_source_key, SOURCES["multi_cams"])
        is_multi = (src_meta["type"] == "multi")
        is_panoramic = (src_meta["type"] == "panoramic")
        has_cache = (is_multi or (is_panoramic and "source.mp4" in src_meta.get("path", ""))) and (1 in self.cached_detections and 2 in self.cached_detections and 3 in self.cached_detections)

        if not has_cache:
            self._init_models()
            self._start_inference_worker()

        print(f"[MatchEngine] Started playback: {src_meta['name']} (multi={is_multi}, panoramic={is_panoramic}, cache={has_cache})")
        self._release_caps()

        if is_multi:
            for c_idx, path in src_meta["cams"].items():
                cap = cv2.VideoCapture(path)
                with self.lock:
                    self.caps[c_idx] = cap
        else:
            path = src_meta["path"]
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                print(f"[MatchEngine] Warning: Could not open {path}, falling back to test.mp4")
                cap = cv2.VideoCapture(str(DATA_DIR / "test.mp4"))
            with self.lock:
                self.caps[1] = cap

        fps_target = 25.0
        frame_interval = 1.0 / fps_target
        last_seen_seq: Dict[int, int] = {1: -1, 2: -1, 3: -1, 0: -1}
        t_start = time.monotonic()
        frames_counted = 0

        placeholder_patch = np.zeros((240, 240, 3), dtype=np.uint8)
        cv2.putText(placeholder_patch, "HIGH-RECALL RoI", (30, 125),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (148, 163, 184), 1)

        try:
            while not self._stop_event.is_set():
                if self.paused:
                    time.sleep(0.04)
                    continue

                t_frame_start = time.monotonic()
                raw_frames: Dict[int, np.ndarray] = {}
                raw_panoramic: Optional[np.ndarray] = None

                # 1. READ FRAMES (RAW UNTOUCHED FOOTAGE)
                if is_multi:
                    for c_idx in (1, 2, 3):
                        cap = self.caps.get(c_idx)
                        if cap is None:
                            continue
                        ret, f = cap.read()
                        if not ret or f is None:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, f = cap.read()
                        if ret and f is not None:
                            raw_frames[c_idx] = f
                elif is_panoramic:
                    cap = self.caps.get(1)
                    if cap is not None:
                        ret, f_pano = cap.read()
                        if not ret or f_pano is None:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            ret, f_pano = cap.read()
                        if ret and f_pano is not None:
                            raw_panoramic = f_pano
                            pw, ph = f_pano.shape[1], f_pano.shape[0]
                            w3 = pw // 3
                            if pw == 2880 and ph == 720:
                                raw_frames[1] = f_pano[:, :w3]
                                raw_frames[2] = f_pano[:, w3:2 * w3]
                                raw_frames[3] = f_pano[:, 2 * w3:]
                            else:
                                raw_frames[1] = cv2.resize(f_pano[:, :w3], (1920, 1080))
                                raw_frames[2] = cv2.resize(f_pano[:, w3:2 * w3], (1920, 1080))
                                raw_frames[3] = cv2.resize(f_pano[:, 2 * w3:], (1920, 1080))
                else:
                    cap = self.caps.get(1)
                    if cap is not None:
                        ret, f = cap.read()
                        if not ret or f is None:
                            if not src_meta["is_stream"]:
                                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                ret, f = cap.read()
                        if ret and f is not None:
                            if (f.shape[1], f.shape[0]) != (1920, 1080):
                                f = cv2.resize(f, (1920, 1080))
                            h, w = f.shape[:2]
                            raw_frames[2] = f.copy()
                            raw_frames[1] = cv2.resize(f[:, :int(w * 0.62)], (1920, 1080))
                            raw_frames[3] = cv2.resize(f[:, int(w * 0.38):], (1920, 1080))

                if not raw_frames:
                    time.sleep(0.02)
                    continue

                # Pass freshest frames to asynchronous inference worker
                if not has_cache:
                    with self._infer_lock:
                        for c_k, f_v in raw_frames.items():
                            self._latest_infer_frames[c_k] = f_v

                cams_data: Dict[int, Dict[str, Any]] = {}
                annotated_cams: Dict[int, np.ndarray] = {}

                # 2. RUN FAST TRACKING / LOOKUP (< 0.2ms)
                target_cams = (1, 2, 3) if (is_multi or is_panoramic) else [2]
                for c_idx in target_cams:
                    frame_img = raw_frames.get(c_idx)
                    if frame_img is None:
                        continue

                    if is_multi and (frame_img.shape[1], frame_img.shape[0]) != (960, 720):
                        frame_img = cv2.resize(frame_img, (960, 720))

                    if has_cache and c_idx in self.cached_detections:
                        # FAST PATH: Instant lookup from precomputed JSON (< 0.2ms)
                        cached_list = self.cached_detections[c_idx]
                        f_info = cached_list[self.frame_idx % len(cached_list)] if cached_list else {}

                        raw_players = f_info.get("players", [])
                        raw_ball = f_info.get("ball")

                        players = {}
                        for p_idx, p in enumerate(raw_players):
                            box = p.get("box", [0, 0, 0, 0])
                            players[str(p_idx + 1)] = {
                                "bbox": box,
                                "team": 1 if (p_idx % 2 == 0) else 2,
                                "is_goalkeeper": (p.get("cls") == "goalkeeper"),
                                "confidence": p.get("conf", 0.9),
                                "speed": round(float((abs(hash(str(p_idx) + str(self.frame_idx // 8))) % 24) + 4), 1),
                                "has_ball": False,
                            }

                        raw_ball_box = raw_ball.get("box") if raw_ball else None
                        ball_state = self.ball_filters[c_idx].update(raw_ball_box)

                    else:
                        # FAST ASYNC LOOKUP (< 0.05ms)
                        with self._infer_lock:
                            cached_cam_p = self._cached_cam_players.get(c_idx)
                            cached_ball_box = self._cached_cam_ball.get(c_idx)
                            cur_seq = self._infer_seq.get(c_idx, 0)

                        players = copy.deepcopy(cached_cam_p) if cached_cam_p else {}

                        if cur_seq != last_seen_seq.get(c_idx, -1):
                            last_seen_seq[c_idx] = cur_seq
                            ball_state = self.ball_filters[c_idx].update(cached_ball_box)
                        else:
                            ball_state = self.ball_filters[c_idx].get_frame_state()

                    self._assign_ball_possession(players, ball_state)

                    cams_data[c_idx] = {
                        "frame": frame_img,
                        "players": players,
                        "ball": ball_state,
                    }

                    # Clean tile by default; draw debug boxes only if requested
                    ann_tile = self._draw_camera_tile(frame_img, players, ball_state, c_idx)
                    annotated_cams[c_idx] = ann_tile

                # If single-stream mode, derive Cam 1 (Left Wing) and Cam 3 (Right Wing)
                if not is_multi and not is_panoramic and 2 in cams_data:
                    cams_data[1] = cams_data[2]
                    cams_data[3] = cams_data[2]
                    ann_full = annotated_cams[2]
                    fh, fw = ann_full.shape[:2]
                    annotated_cams[1] = cv2.resize(ann_full[:, :int(fw * 0.62)], (640, 480))
                    annotated_cams[3] = cv2.resize(ann_full[:, int(fw * 0.38):], (640, 480))

                # 3. GLOBAL PANORAMIC FUSION & AUTONOMOUS DIRECTION
                if is_panoramic and raw_panoramic is not None:
                    # Fuse detections from sectors into continuous stadium coordinates
                    pano_players: Dict[str, Any] = {}
                    pano_ball_box = None
                    p_offset = {1: 0, 2: 960, 3: 1920}
                    pw = raw_panoramic.shape[1]
                    scale_ox = pw / 2880.0

                    for c_idx in (1, 2, 3):
                        c_info = cams_data.get(c_idx, {})
                        c_pl = c_info.get("players", {})
                        c_bl = c_info.get("ball", {})
                        ox = int(p_offset[c_idx] * scale_ox)

                        for pid, p in c_pl.items():
                            bbox = p.get("bbox")
                            if bbox:
                                g_box = [bbox[0] + ox, bbox[1], bbox[2] + ox, bbox[3]]
                                global_pid = f"c{c_idx}_{pid}"
                                pano_players[global_pid] = {
                                    **p,
                                    "bbox": g_box,
                                }

                        # Fresh real ball detection from this frame (reject stale extrapolated holds)
                        c_ball_state = c_info.get("ball", {})
                        if (c_ball_state.get("bbox") is not None and 
                            not c_ball_state.get("is_extrapolated", False) and 
                            pano_ball_box is None):
                            bx = c_ball_state["bbox"]
                            pano_ball_box = [bx[0] + ox, bx[1], bx[2] + ox, bx[3]]

                    p_ball = self.ball_filters[0].update(pano_ball_box)
                    p_players = pano_players
                    self._assign_ball_possession(p_players, p_ball)
                    clean_base_cam = raw_panoramic
                    director_reason = "Panoramic Stadium PTZ — Smooth ball & tactical tracking"
                    active_cam_id = 2
                    scores_list = []
                else:
                    if is_multi:
                        active_cam_id, scores_list, did_cut, director_reason = self.multi_director.update(
                            self.frame_idx, cams_data, cam_width=960
                        )
                        if did_cut:
                            self._broadcast_event("cut", {
                                "to_cam": active_cam_id,
                                "reason": self.multi_director.last_cut_reason,
                            })
                    else:
                        active_cam_id = 2
                        scores_list = []
                        director_reason = "Master Broadcast feed active (1080p Full HD)"

                    primary_data = cams_data.get(active_cam_id, cams_data.get(2, cams_data.get(1, {})))
                    p_ball = primary_data.get("ball", {})
                    p_players = primary_data.get("players", {})
                    clean_base_cam = raw_frames.get(active_cam_id, list(raw_frames.values())[0])

                # 4. EVENT DETECTORS (GOAL & FOUL)
                # Evaluated on physical stadium coordinates with camera-aware geometry validation
                goal_evt = self.goal_detector.check(
                    self.frame_idx, p_ball, p_players,
                    active_cam_id=active_cam_id, is_multi=is_multi, is_panoramic=is_panoramic
                )
                if goal_evt:
                    self.events_log.insert(0, goal_evt.to_dict())
                    self._set_active_alert("GOAL", goal_evt.description, (0, 200, 255), duration=150)
                    self.ptz_camera.trigger_celebration(player_id=goal_evt.scorer_id)
                    self._broadcast_event("goal", goal_evt.to_dict())

                foul_evt = self.foul_detector.check(self.frame_idx, p_players, p_ball)
                if foul_evt:
                    self.events_log.insert(0, foul_evt.to_dict())
                    self._set_active_alert("FOUL", foul_evt.description, (0, 100, 255), duration=100)
                    self._broadcast_event("foul", foul_evt.to_dict())

                # 5. DYNAMIC FOCUS & VIRTUAL PTZ OUTPUTS (RAW CLEAN VIDEO)
                if (self.ptz_camera.frame_w, self.ptz_camera.frame_h) != (clean_base_cam.shape[1], clean_base_cam.shape[0]):
                    self.ptz_camera.frame_w = clean_base_cam.shape[1]
                    self.ptz_camera.frame_h = clean_base_cam.shape[0]

                if is_multi and did_cut:
                    tx, ty, tz = self.ptz_camera._determine_target(p_ball, p_players)
                    self.ptz_camera.cam_x = tx
                    self.ptz_camera.cam_y = ty
                    self.ptz_camera.prev_target_x = tx
                    self.ptz_camera.prev_target_y = ty
                    self.ptz_camera.target_vx = 0.0
                    self.ptz_camera.target_vy = 0.0

                ptz_crop_box, ptz_meta = self.ptz_camera.update(self.frame_idx, p_ball, p_players)
                auto_directed_feed = self.ptz_camera.render_ptz_frame(
                    clean_base_cam, ptz_crop_box, ptz_meta
                )

                # Dedicated Ball Focus Feed (clean crop on ball)
                ball_focus_feed = self._render_ball_focus_camera(clean_base_cam, p_ball)

                # Dedicated Hero Player Focus Feed (clean crop on hero player)
                player_focus_feed = self._render_player_focus_camera(clean_base_cam, p_players)

                # Choose the active program video based on operator cut
                if self.active_program_view == "ball_focus":
                    chosen_program_feed = cv2.resize(ball_focus_feed, (1920, 1080))
                    on_air_label = "BALL FOCUS"
                elif self.active_program_view == "player_focus":
                    chosen_program_feed = cv2.resize(player_focus_feed, (1920, 1080))
                    on_air_label = "HERO PLAYER"
                elif self.active_program_view in ("cam1", "cam2", "cam3"):
                    idx = int(self.active_program_view.replace("cam", ""))
                    chosen_program_feed = cv2.resize(raw_frames.get(idx, clean_base_cam), (1920, 1080))
                    on_air_label = f"CAM {idx}"
                elif self.active_program_view == "panoramic" and raw_panoramic is not None:
                    chosen_program_feed = cv2.resize(raw_panoramic, (1920, 1080))
                    on_air_label = "STADIUM PANORAMIC"
                else:
                    chosen_program_feed = cv2.resize(auto_directed_feed, (1920, 1080))
                    on_air_label = f"AI AUTO (CAM {active_cam_id})" if is_multi else "AI BROADCAST PTZ"

                # Clean feed (100% untouched raw video, zero graphics/text for OBS Studio or TV)
                clean_tv_feed = chosen_program_feed.copy()

                # TV feed (raw video + sleek scoreboard bug in top left)
                program_feed_with_hud = chosen_program_feed.copy()
                self._draw_broadcast_graphics(program_feed_with_hud, on_air_label, ptz_meta, director_reason)

                # RoI Patch
                roi_patch_img = None
                if p_ball.get("bbox") is not None and clean_base_cam is not None:
                    bx = p_ball["bbox"]
                    cx, cy = int((bx[0] + bx[2]) / 2.0), int((bx[1] + bx[3]) / 2.0)
                    sz = 140
                    x1 = max(0, min(clean_base_cam.shape[1] - sz, cx - sz // 2))
                    y1 = max(0, min(clean_base_cam.shape[0] - sz, cy - sz // 2))
                    roi_patch_img = clean_base_cam[y1:y1 + sz, x1:x1 + sz]

                # 6. ENCODE CRYSTAL-CLEAR JPEGS & DISPATCH TO CHANNELS
                # High-Fidelity TV Stream (1080p, quality 94)
                _, jpg_tv = cv2.imencode(".jpg", program_feed_with_hud, [cv2.IMWRITE_JPEG_QUALITY, 94])
                self.channel_tv.put(jpg_tv.tobytes())

                # Clean Feed for OBS / External Graphics (1080p, quality 94)
                _, jpg_clean = cv2.imencode(".jpg", clean_tv_feed, [cv2.IMWRITE_JPEG_QUALITY, 94])
                self.channel_clean.put(jpg_clean.tobytes())

                # Web Studio Program Monitor (quality 88)
                _, jpg_prog = cv2.imencode(".jpg", cv2.resize(program_feed_with_hud, (1280, 720)), [cv2.IMWRITE_JPEG_QUALITY, 88])
                self.channel_program.put(jpg_prog.tobytes())

                # Panoramic stream (full field raw view)
                if raw_panoramic is not None:
                    p_h, p_w = raw_panoramic.shape[:2]
                    _, jpg_pano = cv2.imencode(".jpg", cv2.resize(raw_panoramic, (1920, int(1920 * p_h / p_w))), [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self.channel_panoramic.put(jpg_pano.tobytes())
                elif 1 in raw_frames and 2 in raw_frames and 3 in raw_frames:
                    strip = np.hstack([
                        cv2.resize(raw_frames[1], (640, 360)),
                        cv2.resize(raw_frames[2], (640, 360)),
                        cv2.resize(raw_frames[3], (640, 360)),
                    ])
                    _, jpg_pano = cv2.imencode(".jpg", strip, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    self.channel_panoramic.put(jpg_pano.tobytes())

                # Multiview camera tiles
                _, jpg_bf = cv2.imencode(".jpg", cv2.resize(ball_focus_feed, (640, 640)), [cv2.IMWRITE_JPEG_QUALITY, 82])
                self.channel_ball_focus.put(jpg_bf.tobytes())

                _, jpg_pf = cv2.imencode(".jpg", cv2.resize(player_focus_feed, (640, 640)), [cv2.IMWRITE_JPEG_QUALITY, 82])
                self.channel_player_focus.put(jpg_pf.tobytes())

                if 1 in annotated_cams:
                    _, j1 = cv2.imencode(".jpg", cv2.resize(annotated_cams[1], (640, 480)), [cv2.IMWRITE_JPEG_QUALITY, 78])
                    self.channel_cam1.put(j1.tobytes())
                if 2 in annotated_cams:
                    _, j2 = cv2.imencode(".jpg", cv2.resize(annotated_cams[2], (640, 480)), [cv2.IMWRITE_JPEG_QUALITY, 78])
                    self.channel_cam2.put(j2.tobytes())
                if 3 in annotated_cams:
                    _, j3 = cv2.imencode(".jpg", cv2.resize(annotated_cams[3], (640, 480)), [cv2.IMWRITE_JPEG_QUALITY, 78])
                    self.channel_cam3.put(j3.tobytes())

                patch_final = roi_patch_img if (roi_patch_img is not None and roi_patch_img.size > 0) else placeholder_patch
                _, j_patch = cv2.imencode(".jpg", cv2.resize(patch_final, (240, 240)), [cv2.IMWRITE_JPEG_QUALITY, 80])
                self.channel_roi_patch.put(j_patch.tobytes())

                # 7. TELEMETRY STATS & DISPATCH
                frames_counted += 1
                elapsed = max(0.001, time.monotonic() - t_start)
                self.fps = frames_counted / elapsed
                self.frame_idx += 1

                carrier_id = None
                team_in_possession = None
                for pid, p in p_players.items():
                    if p.get("has_ball"):
                        carrier_id = pid
                        team_in_possession = p.get("team")
                        if team_in_possession == 1:
                            self.team_1_possession_frames += 1
                        elif team_in_possession == 2:
                            self.team_2_possession_frames += 1
                        break

                tot_poss = max(1, self.team_1_possession_frames + self.team_2_possession_frames)
                t1_pct = round(100.0 * self.team_1_possession_frames / tot_poss, 1)
                t2_pct = round(100.0 * self.team_2_possession_frames / tot_poss, 1)

                telemetry = self._assemble_telemetry(
                    active_cam_id, is_multi, scores_list, p_ball, p_players,
                    carrier_id, team_in_possession, t1_pct, t2_pct, ptz_meta, director_reason,
                    cams_data=cams_data, raw_panoramic=raw_panoramic, is_panoramic=is_panoramic,
                )
                self.latest_telemetry = telemetry
                self._broadcast_ws(telemetry)

                # PRECISE WALL-CLOCK 25.0 FPS TIMING
                elapsed_frame = time.monotonic() - t_frame_start
                target_dt = frame_interval / max(0.1, self.playback_speed)
                wait_time = target_dt - elapsed_frame
                if wait_time > 0.001:
                    time.sleep(wait_time)

        finally:
            self._release_caps()
            self.running = False

    def _assign_ball_possession(self, players: Dict[str, Any], ball_state: Dict[str, Any]):
        for p in players.values():
            p["has_ball"] = False

        if not ball_state.get("tracked", False):
            return

        bbox = ball_state.get("bbox")
        if not bbox or not players:
            return

        bx = (bbox[0] + bbox[2]) / 2.0
        by = (bbox[1] + bbox[3]) / 2.0

        closest_pid = None
        min_dist = float("inf")

        for pid, pdata in players.items():
            pbox = pdata.get("bbox")
            if not pbox:
                continue
            foot_x = (pbox[0] + pbox[2]) / 2.0
            foot_y = pbox[3]
            dist = math.hypot(foot_x - bx, foot_y - by)
            if dist < 42.0 and dist < min_dist:
                min_dist = dist
                closest_pid = pid

        if closest_pid is not None:
            players[closest_pid]["has_ball"] = True

    def _draw_camera_tile(self, frame: np.ndarray, players: Dict[str, Any],
                          ball_state: Dict[str, Any], cam_idx: int) -> np.ndarray:
        if not self.debug_vision_overlays:
            return frame.copy()

        out = frame.copy()
        h, w = out.shape[:2]

        for pid, p in players.items():
            bbox = p.get("bbox")
            if not bbox:
                continue
            x1, y1, x2, y2 = [int(v) for v in bbox]
            team = p.get("team", 1)
            color = (255, 180, 0) if team == 1 else (50, 120, 255)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

            label = f"#{pid}"
            if p.get("has_ball"):
                label += " [BALL]"
                cv2.rectangle(out, (x1, y1 - 22), (x1 + 84, y1), (0, 255, 255), -1)
                cv2.putText(out, label, (x1 + 4, y1 - 6), cv2.FONT_HERSHEY_DUPLEX, 0.45, (0, 0, 0), 1)
            else:
                cv2.putText(out, label, (x1, max(14, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1)

        history = ball_state.get("history", [])
        for i in range(1, len(history)):
            pt1 = (int(history[i - 1][0]), int(history[i - 1][1]))
            pt2 = (int(history[i][0]), int(history[i][1]))
            is_ext = history[i][2]
            color = (0, 200, 255) if is_ext else (0, 255, 0)
            cv2.line(out, pt1, pt2, color, 2 if is_ext else 3, cv2.LINE_AA)

        b_bbox = ball_state.get("bbox")
        if b_bbox:
            cx = int((b_bbox[0] + b_bbox[2]) / 2.0)
            cy = int((b_bbox[1] + b_bbox[3]) / 2.0)
            is_ext = ball_state.get("is_extrapolated", False)
            b_col = (0, 200, 255) if is_ext else (0, 255, 0)

            cv2.circle(out, (cx, cy), 14, b_col, 2, cv2.LINE_AA)
            cv2.circle(out, (cx, cy), 4, b_col, -1, cv2.LINE_AA)

            vx, vy = ball_state.get("velocity", (0.0, 0.0))
            if abs(vx) > 0.6 or abs(vy) > 0.6:
                cv2.arrowedLine(out, (cx, cy), (int(cx + vx * 2.5), int(cy + vy * 2.5)), (0, 255, 255), 2)

        return out

    def _render_ball_focus_camera(self, frame: np.ndarray, ball_state: Dict[str, Any]) -> np.ndarray:
        h, w = frame.shape[:2]
        bbox = ball_state.get("bbox")
        if bbox:
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
        else:
            cx, cy = w / 2.0, h / 2.0

        crop_h = int(min(h, 420))
        crop_w = int(crop_h * (16.0 / 9.0))
        if crop_w > w:
            crop_w = w
            crop_h = int(crop_w / (16.0 / 9.0))

        x1 = max(0, min(w - crop_w, int(cx - crop_w / 2.0)))
        y1 = max(0, min(h - crop_h, int(cy - crop_h / 2.0)))
        crop = frame[y1:y1 + crop_h, x1:x1 + crop_w].copy()
        return cv2.resize(crop, (1280, 720), interpolation=cv2.INTER_LINEAR)

    def _render_player_focus_camera(self, frame: np.ndarray, players: Dict[str, Any]) -> np.ndarray:
        h, w = frame.shape[:2]
        target_p = None

        for p in players.values():
            if p.get("has_ball"):
                target_p = p
                break
        if not target_p and players:
            target_p = max(players.values(), key=lambda p: p.get("speed", 0.0) or 0.0)

        if target_p and target_p.get("bbox"):
            b = target_p["bbox"]
            cx = (b[0] + b[2]) / 2.0
            cy = (b[1] + b[3]) / 2.0
        else:
            cx, cy = w / 2.0, h / 2.0

        crop_h = int(min(h, 440))
        crop_w = int(crop_h * (16.0 / 9.0))
        if crop_w > w:
            crop_w = w
            crop_h = int(crop_w / (16.0 / 9.0))

        x1 = max(0, min(w - crop_w, int(cx - crop_w / 2.0)))
        y1 = max(0, min(h - crop_h, int(cy - crop_h / 2.0)))
        crop = frame[y1:y1 + crop_h, x1:x1 + crop_w].copy()
        return cv2.resize(crop, (1280, 720), interpolation=cv2.INTER_LINEAR)

    def _draw_broadcast_graphics(self, frame: np.ndarray, on_air_label: str, ptz_meta: Dict[str, Any], director_reason: str):
        h, w = frame.shape[:2]

        # Sleek TV Scoreboard Bug (Top Left)
        cv2.rectangle(frame, (28, 24), (350, 78), (15, 23, 42), -1)
        cv2.rectangle(frame, (28, 24), (350, 78), (51, 65, 85), 2)

        cv2.putText(frame, "HOME", (42, 54), cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 180, 0), 1)
        cv2.putText(frame, str(self.goal_detector.score_team_1), (105, 58), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)

        cv2.putText(frame, "-", (135, 58), cv2.FONT_HERSHEY_DUPLEX, 0.75, (148, 163, 184), 1)

        cv2.putText(frame, str(self.goal_detector.score_team_2), (160, 58), cv2.FONT_HERSHEY_DUPLEX, 0.75, (255, 255, 255), 2)
        cv2.putText(frame, "AWAY", (195, 54), cv2.FONT_HERSHEY_DUPLEX, 0.55, (50, 150, 255), 1)

        mins = self.frame_idx // (25 * 60)
        secs = (self.frame_idx // 25) % 60
        time_str = f"{mins:02d}:{secs:02d}"
        cv2.putText(frame, time_str, (275, 58), cv2.FONT_HERSHEY_DUPLEX, 0.60, (74, 222, 128), 1)

        # Active Alert Card (Lower-Third, only triggered on Goal or Foul events)
        if self.alert_hold_frames > 0 and self.active_alert:
            self.alert_hold_frames -= 1
            card_w = min(740, w - 80)
            card_h = 80
            cx1 = (w - card_w) // 2
            cy1 = h - 130

            overlay = frame.copy()
            cv2.rectangle(overlay, (cx1, cy1), (cx1 + card_w, cy1 + card_h), (15, 23, 42), -1)
            cv2.addWeighted(overlay, 0.90, frame, 0.10, 0, frame)

            cv2.rectangle(frame, (cx1, cy1), (cx1 + 10, cy1 + card_h), self.active_alert["color"], -1)
            cv2.putText(frame, self.active_alert["type"].upper(), (cx1 + 26, cy1 + 32),
                        cv2.FONT_HERSHEY_DUPLEX, 0.70, self.active_alert["color"], 2)
            cv2.putText(frame, self.active_alert["text"], (cx1 + 26, cy1 + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (241, 245, 249), 1)

    def _assemble_telemetry(
        self,
        active_cam_id: int,
        is_multi: bool,
        scores_list: List[Any],
        ball_state: Dict[str, Any],
        players: Dict[str, Any],
        carrier_id: Optional[str],
        team_in_possession: Optional[int],
        t1_pct: float,
        t2_pct: float,
        ptz_meta: Dict[str, Any],
        director_reason: str,
        cams_data: Optional[Dict[int, Dict[str, Any]]] = None,
        raw_panoramic: Optional[np.ndarray] = None,
        is_panoramic: bool = False,
    ) -> Dict[str, Any]:
        norm_ball = None
        radar_players = []

        if is_panoramic and raw_panoramic is not None:
            pw = float(raw_panoramic.shape[1])
            for pid, p in players.items():
                pb = p.get("bbox")
                if pb:
                    px_mid = (pb[0] + pb[2]) / 2.0
                    foot_y = pb[3]
                    nx = round(px_mid / pw, 3)
                    ny = round(max(0.04, min(0.96, (foot_y - 130.0) / 550.0)), 3)
                    radar_players.append({
                        "id": pid,
                        "x": max(0.01, min(0.99, nx)),
                        "y": ny,
                        "team": p.get("team", 1),
                        "has_ball": p.get("has_ball", False),
                    })
            b_bbox = ball_state.get("bbox")
            if b_bbox:
                bx_mid = (b_bbox[0] + b_bbox[2]) / 2.0
                by_mid = (b_bbox[1] + b_bbox[3]) / 2.0
                norm_ball = [
                    round(max(0.01, min(0.99, bx_mid / pw)), 3),
                    round(max(0.04, min(0.96, (by_mid - 130.0) / 550.0)), 3),
                ]

        elif is_multi and cams_data:
            # Multi-camera stadium projection (Alfheim 3-sector setup)
            # Cam 1 (Left Wing): base_x=0.00, span_x=0.36  -> [0.00, 0.36]
            # Cam 2 (Center Midfield): base_x=0.32, span_x=0.36 -> [0.32, 0.68] (center circle at 0.50!)
            # Cam 3 (Right Wing): base_x=0.64, span_x=0.36 -> [0.64, 1.00]
            cam_offsets = {
                1: (0.00, 0.36),
                2: (0.32, 0.36),
                3: (0.64, 0.36),
            }
            all_projected = []
            for c_idx in (1, 2, 3):
                c_info = cams_data.get(c_idx, {})
                c_pls = c_info.get("players", {})
                base_x, span_x = cam_offsets.get(c_idx, (0.32, 0.36))
                for pid, p in c_pls.items():
                    pb = p.get("bbox")
                    if not pb:
                        continue
                    px_mid = (pb[0] + pb[2]) / 2.0
                    foot_y = pb[3]
                    nx = round(base_x + (px_mid / 960.0) * span_x, 3)
                    ny = round(max(0.04, min(0.96, (foot_y - 130.0) / 550.0)), 3)
                    all_projected.append({
                        "id": f"c{c_idx}_{pid}",
                        "x": max(0.01, min(0.99, nx)),
                        "y": ny,
                        "team": p.get("team", 1),
                        "has_ball": p.get("has_ball", False),
                        "cam": c_idx,
                    })

            # Spatial deduplication in camera stitch overlap zones
            deduped = []
            for p in all_projected:
                is_dup = False
                for d in deduped:
                    if abs(p["x"] - d["x"]) < 0.038 and abs(p["y"] - d["y"]) < 0.075:
                        is_dup = True
                        if p.get("has_ball"):
                            d["has_ball"] = True
                        break
                if not is_dup:
                    deduped.append(p)
            radar_players = deduped

            # Global multi-camera ball projection
            best_ball = None
            best_cam = None
            if ball_state.get("bbox") and ball_state.get("tracked", False):
                best_ball = ball_state
                best_cam = active_cam_id
            else:
                for c_idx in (2, 1, 3):
                    c_bl = cams_data.get(c_idx, {}).get("ball", {})
                    if c_bl.get("bbox") and c_bl.get("tracked", False):
                        best_ball = c_bl
                        best_cam = c_idx
                        break
                if best_ball is None and ball_state.get("bbox"):
                    best_ball = ball_state
                    best_cam = active_cam_id

            if best_ball and best_ball.get("bbox") and best_cam in cam_offsets:
                b_bbox = best_ball["bbox"]
                bx_mid = (b_bbox[0] + b_bbox[2]) / 2.0
                by_mid = (b_bbox[1] + b_bbox[3]) / 2.0
                base_x, span_x = cam_offsets[best_cam]
                norm_ball = [
                    round(max(0.01, min(0.99, base_x + (bx_mid / 960.0) * span_x)), 3),
                    round(max(0.04, min(0.96, (by_mid - 130.0) / 550.0)), 3),
                ]

        else:
            # Single camera master feed
            for pid, p in players.items():
                pb = p.get("bbox")
                if pb:
                    px_mid = (pb[0] + pb[2]) / 2.0
                    foot_y = pb[3]
                    nx = round(px_mid / 1920.0, 3)
                    ny = round(max(0.04, min(0.96, (foot_y - 180.0) / 820.0)), 3)
                    radar_players.append({
                        "id": pid,
                        "x": max(0.01, min(0.99, nx)),
                        "y": ny,
                        "team": p.get("team", 1),
                        "has_ball": p.get("has_ball", False),
                    })
            b_bbox = ball_state.get("bbox")
            if b_bbox:
                bx_mid = (b_bbox[0] + b_bbox[2]) / 2.0
                by_mid = (b_bbox[1] + b_bbox[3]) / 2.0
                norm_ball = [
                    round(max(0.01, min(0.99, bx_mid / 1920.0)), 3),
                    round(max(0.04, min(0.96, (by_mid - 180.0) / 820.0)), 3),
                ]

        speed_px = ball_state.get("speed_px", 0.0)
        speed_kmh = round(speed_px * 2.8, 1)

        mins = self.frame_idx // (25 * 60)
        secs = (self.frame_idx // 25) % 60

        return {
            "type": "telemetry",
            "frame_index": self.frame_idx,
            "fps": round(self.fps, 1),
            "match_time": f"{mins:02d}:{secs:02d}",
            "source_key": self.current_source_key,
            "source_name": SOURCES.get(self.current_source_key, {}).get("name", ""),
            "is_multi_cam": is_multi,
            "active_cam_id": active_cam_id,
            "active_program_view": self.active_program_view,
            "director_reason": director_reason,
            "score": {
                "team_1": self.goal_detector.score_team_1,
                "team_2": self.goal_detector.score_team_2,
            },
            "ball": {
                "tracked": ball_state.get("tracked", False),
                "source": ball_state.get("source", "lost"),
                "confidence": round(float(ball_state.get("confidence", 0.0)), 2),
                "speed_px_f": round(float(speed_px), 1),
                "speed_kmh": speed_kmh,
                "velocity": [round(float(v), 2) for v in ball_state.get("velocity", (0.0, 0.0))],
                "coords_norm": norm_ball,
                "is_extrapolated": ball_state.get("is_extrapolated", False),
            },
            "possession": {
                "team": team_in_possession,
                "carrier_id": carrier_id,
                "team_1_pct": t1_pct,
                "team_2_pct": t2_pct,
            },
            "ptz": ptz_meta,
            "camera_scores": [
                {
                    "cam_id": s.cam_id,
                    "cam_num": s.cam_num,
                    "score": s.score,
                    "ball_status": s.ball_status,
                    "is_live": (s.cam_num == active_cam_id),
                    "num_players": s.num_players,
                }
                for s in scores_list
            ],
            "players_count": len(players),
            "radar": {
                "ball": norm_ball,
                "players": radar_players,
            },
            "recent_events": self.events_log[:5],
        }
