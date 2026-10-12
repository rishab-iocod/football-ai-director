"""High-Recall Hybrid Ball Tracker.

Combines three complementary layers to solve the 4/10 missed-frame problem:
1. Dynamic Native-Resolution Temporal RoI (recovering tiny balls lost to downscaling)
2. Kinematic Trajectory Consistency Validator (filtering socks/grass noise)
3. Physics-Constrained Flight Kalman Filter (bridging occlusions and multi-frame gaps)
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
from ultralytics import YOLO

from .physics_kalman import PhysicsBallKalman
from .temporal_roi import TemporalRoICropper
from .trajectory_validator import TrajectoryValidator


class HighRecallBallTracker:
    """High-recall ball tracking engine for football broadcast streams."""

    def __init__(self,
                 model_path="models/best.pt",
                 ball_class_id=0,
                 device=None,
                 half=False,
                 base_conf=0.12,
                 patch_conf=0.10,
                 frame_size=(1920, 1080),
                 max_extrapolate_frames=12):
        self.model = YOLO(model_path)
        if device is not None:
            self.model.to(device)
        self.device = device
        self.half = half
        self.ball_class_id = ball_class_id

        self.base_conf = base_conf
        self.patch_conf = patch_conf
        self.frame_size = frame_size

        # Core tracking engines
        self.kalman = PhysicsBallKalman(max_extrapolate_frames=max_extrapolate_frames)
        self.cropper = TemporalRoICropper(frame_size=frame_size)
        self.validator = TrajectoryValidator()

        # Operational metrics
        self.total_frames = 0
        self.detected_frames = 0
        self.extrapolated_frames = 0

    def reset(self):
        """Reset state on camera cut or prolonged play stoppage."""
        self.kalman.reset()

    def track(self, frame, global_detections=None):
        """Process one frame and return high-recall ball state telemetry.

        Parameters:
            frame: (H, W, 3) BGR image.
            global_detections: Optional precomputed YOLO results or detections dict
                               to avoid redundant global inference.
        """
        self.total_frames += 1

        # 1. Physics prediction
        predicted_pos = self.kalman.predict() if self.kalman.is_initialized else None
        search_radius = self.kalman.get_search_radius() if self.kalman.is_initialized else None

        best_cand = None
        detection_source = None

        # 2. STRATEGY 1: Dynamic Native-Resolution RoI Patch
        # If ball is currently tracked, inspect the high-resolution patch first
        if predicted_pos is not None:
            crop_box = self.cropper.compute_crop_box(predicted_pos, search_radius)
            patch = self.cropper.extract_patch(frame, crop_box)

            if patch.shape[0] > 32 and patch.shape[1] > 32:
                patch_candidates = self._detect_in_patch(patch, crop_box)
                best_cand = self.validator.rank_candidates(
                    patch_candidates, predicted_pos, search_radius)
                if best_cand is not None:
                    detection_source = "temporal_roi_patch"

        # 3. STRATEGY 2: Global Frame Fallback
        # If patch did not find the ball (or first frame), run/use global detections
        if best_cand is None:
            if global_detections is not None:
                global_candidates = self._extract_ball_candidates_from_precomputed(global_detections)
            else:
                global_candidates = self._detect_global(frame)

            best_cand = self.validator.rank_candidates(
                global_candidates, predicted_pos, search_radius)
            if best_cand is not None:
                detection_source = "global_frame"

        # 4. Filter Update / Trajectory Bridging
        if best_cand is not None:
            bbox = best_cand["bbox"]
            conf = best_cand.get("conf", 0.5)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0

            updated_x, updated_y = self.kalman.update(cx, cy, w, h)
            self.detected_frames += 1
            vx, vy, speed = self.kalman.get_velocity()

            return {
                "tracked": True,
                "bbox": [updated_x - w / 2.0, updated_y - h / 2.0,
                         updated_x + w / 2.0, updated_y + h / 2.0],
                "center": (updated_x, updated_y),
                "velocity": (vx, vy),
                "speed_px": speed,
                "is_extrapolated": False,
                "missed_frames": 0,
                "confidence": conf,
                "source": detection_source,
                "history": list(self.kalman.history),
            }

        # 5. STRATEGY 3: Physics Flight Extrapolation (Bridging the gap)
        extrapolated = self.kalman.handle_missed_frame()
        if extrapolated is not None:
            self.extrapolated_frames += 1
            bbox = self.kalman.get_current_bbox()
            vx, vy, speed = self.kalman.get_velocity()
            return {
                "tracked": True,
                "bbox": bbox,
                "center": extrapolated,
                "velocity": (vx, vy),
                "speed_px": speed,
                "is_extrapolated": True,
                "missed_frames": self.kalman.missed_frames,
                "confidence": max(0.1, 0.85 - self.kalman.missed_frames * 0.08),
                "source": "physics_extrapolated",
                "history": list(self.kalman.history),
            }

        # Ball is definitively lost
        return {
            "tracked": False,
            "bbox": None,
            "center": None,
            "velocity": (0.0, 0.0),
            "speed_px": 0.0,
            "is_extrapolated": False,
            "missed_frames": self.kalman.missed_frames,
            "confidence": 0.0,
            "source": "lost",
            "history": [],
        }

    def _detect_in_patch(self, patch, crop_box):
        """Run YOLO inference on high-resolution patch and map coordinates to global frame."""
        results = self.model.predict(
            patch,
            conf=self.patch_conf,
            device=self.device,
            half=self.half,
            verbose=False,
            imgsz=patch.shape[0],
        )
        candidates = []
        if not results or len(results) == 0:
            return candidates

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return candidates

        for xyxy, cls_id, conf in zip(boxes.xyxy.cpu().numpy(),
                                      boxes.cls.cpu().numpy(),
                                      boxes.conf.cpu().numpy()):
            if int(cls_id) == self.ball_class_id:
                global_bbox = self.cropper.map_to_global_coords(xyxy, crop_box)
                candidates.append({"bbox": global_bbox, "conf": float(conf)})

        return candidates

    def _detect_global(self, frame):
        """Run standard YOLO inference across full frame."""
        results = self.model.predict(
            frame,
            conf=self.base_conf,
            device=self.device,
            half=self.half,
            verbose=False,
        )
        candidates = []
        if not results or len(results) == 0:
            return candidates

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return candidates

        for xyxy, cls_id, conf in zip(boxes.xyxy.cpu().numpy(),
                                      boxes.cls.cpu().numpy(),
                                      boxes.conf.cpu().numpy()):
            if int(cls_id) == self.ball_class_id:
                candidates.append({
                    "bbox": [float(v) for v in xyxy],
                    "conf": float(conf),
                })

        return candidates

    def _extract_ball_candidates_from_precomputed(self, detections):
        """Helper to extract ball bboxes if precomputed YOLO output is passed."""
        candidates = []
        if isinstance(detections, dict) and "ball" in detections:
            for b_info in detections["ball"].values():
                if "bbox" in b_info:
                    candidates.append({"bbox": b_info["bbox"], "conf": b_info.get("conf", 0.5)})
        return candidates

    @property
    def recall_stats(self):
        """Return recall telemetry metrics."""
        total = max(1, self.total_frames)
        return {
            "total_frames": self.total_frames,
            "detected_frames": self.detected_frames,
            "extrapolated_frames": self.extrapolated_frames,
            "effective_coverage_pct": round(((self.detected_frames + self.extrapolated_frames) / total) * 100.0, 1),
            "raw_detection_pct": round((self.detected_frames / total) * 100.0, 1),
        }
