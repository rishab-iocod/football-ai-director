"""Kinematic Trajectory Consistency Validator.

Rejects false-positive ball detections (socks, boots, white pitch markings, sponsor logos)
by enforcing physical motion constraints and kinematic plausibility.
"""

import math


class TrajectoryValidator:
    """Validates candidate ball bounding boxes against kinematic constraints."""

    def __init__(self,
                 max_jump_px=180.0,
                 min_aspect_ratio=0.55,
                 max_aspect_ratio=1.85,
                 min_ball_dim=6.0,
                 max_ball_dim=65.0):
        self.max_jump = max_jump_px
        self.min_aspect = min_aspect_ratio
        self.max_aspect = max_aspect_ratio
        self.min_dim = min_ball_dim
        self.max_dim = max_ball_dim

    def validate_candidate(self, bbox, conf, predicted_center=None, search_radius=None):
        """Evaluate whether a candidate detection is physically consistent.

        Returns (is_valid, penalty_score). Lower penalty score = better candidate.
        """
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        # 1. Size bounds check
        if w < self.min_dim or h < self.min_dim or w > self.max_dim or h > self.max_dim:
            return False, 999.0

        # 2. Aspect ratio check (circle or motion-blurred ellipse)
        aspect = float(w) / max(1.0, float(h))
        if aspect < self.min_aspect or aspect > self.max_aspect:
            return False, 999.0

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0

        # 3. Kinematic proximity to predicted position
        if predicted_center is not None:
            pred_x, pred_y = predicted_center
            dist = math.hypot(cx - pred_x, cy - pred_y)

            allowed_radius = search_radius if search_radius is not None else self.max_jump
            if dist > allowed_radius:
                return False, 999.0

            # Score combines confidence and distance to physical prediction
            # Higher confidence + smaller displacement = lower penalty
            score = (dist / max(1.0, allowed_radius)) * 0.6 + (1.0 - conf) * 0.4
            return True, score

        # If no prediction exists (initial frame), score purely by confidence and aspect
        aspect_penalty = abs(1.0 - aspect) * 0.2
        score = (1.0 - conf) + aspect_penalty
        return True, score

    def rank_candidates(self, candidates, predicted_center=None, search_radius=None):
        """Rank a list of candidate dicts [{'bbox': ..., 'conf': ...}] and return the best."""
        valid_candidates = []
        for cand in candidates:
            bbox = cand["bbox"]
            conf = cand.get("conf", 0.5)
            is_valid, score = self.validate_candidate(bbox, conf, predicted_center, search_radius)
            if is_valid:
                valid_candidates.append((score, cand))

        if not valid_candidates:
            return None

        # Sort by lowest penalty score
        valid_candidates.sort(key=lambda item: item[0])
        return valid_candidates[0][1]
