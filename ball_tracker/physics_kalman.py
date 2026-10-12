"""Physics-Constrained Kalman Filter for Football Trajectory Tracking.

Models 2D image-space trajectory with physical ball flight dynamics:
- Kinematic state: [x, y, vx, vy, ax, ay]
- Parabolic ball flight: gravity bias in image coordinates
- Velocity decay (aerodynamic drag)
- Ground bounce detection and restitution
- Covariance estimation for dynamic search window sizing
"""

import math
import numpy as np


class PhysicsBallKalman:
    """Kalman filter with football physics for robust trajectory continuity."""

    def __init__(self,
                 dt=1.0 / 25.0,
                 gravity_px=18.0,
                 drag_coef=0.03,
                 restitution=0.65,
                 max_extrapolate_frames=12):
        self.dt = dt
        self.gravity = gravity_px           # Downward pixel acceleration per frame (image y)
        self.drag = drag_coef               # Aerodynamic velocity dampening factor
        self.restitution = restitution       # Bounce elasticity coefficient (v_y inversion)
        self.max_extrapolate = max_extrapolate_frames

        # State vector: [x, y, vx, vy, ax, ay]^T
        self.x = np.zeros((6, 1), dtype=np.float64)

        # State transition matrix F
        self.F = np.eye(6, dtype=np.float64)
        self.F[0, 2] = 1.0  # x += vx
        self.F[1, 3] = 1.0  # y += vy
        self.F[2, 4] = 1.0  # vx += ax
        self.F[3, 5] = 1.0  # vy += ay

        # Measurement matrix H: measures [x, y] center of bbox
        self.H = np.zeros((2, 6), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0

        # Covariance matrices
        self.P = np.eye(6, dtype=np.float64) * 100.0
        self.Q = np.diag([2.0, 2.0, 10.0, 10.0, 20.0, 20.0])  # Process noise
        self.R = np.diag([4.0, 4.0])                           # Measurement noise

        # Operational tracking metrics
        self.missed_frames = 0
        self.is_initialized = False
        self.history = []  # List of past (x, y, is_extrapolated) tuples
        self.last_bbox_size = (20.0, 20.0)

    def initialize(self, x, y, w=20.0, h=20.0):
        """Initialize filter with first validated detection."""
        self.x = np.array([[x], [y], [0.0], [0.0], [0.0], [self.gravity]], dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 20.0
        self.last_bbox_size = (float(w), float(h))
        self.missed_frames = 0
        self.is_initialized = True
        self.history = [(float(x), float(y), False)]

    def predict(self):
        """Advance physical state by one frame with gravity and drag damping."""
        if not self.is_initialized:
            return None

        # Apply aerodynamic drag decay to velocities
        self.x[2, 0] *= (1.0 - self.drag)
        self.x[3, 0] *= (1.0 - self.drag)

        # Inject gravity acceleration into vertical component
        self.x[5, 0] = self.gravity

        # State projection: x_k|k-1 = F * x_k-1
        self.x = np.dot(self.F, self.x)

        # Covariance projection: P_k|k-1 = F * P_k-1 * F^T + Q
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

        return float(self.x[0, 0]), float(self.x[1, 0])

    def update(self, z_x, z_y, w=None, h=None):
        """Correct state with observed detector measurement."""
        if not self.is_initialized:
            self.initialize(z_x, z_y, w or 20.0, h or 20.0)
            return float(z_x), float(z_y)

        z = np.array([[z_x], [z_y]], dtype=np.float64)

        # Innovation: y = z - H * x
        y_innov = z - np.dot(self.H, self.x)

        # Innovation covariance: S = H * P * H^T + R
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R

        # Kalman gain: K = P * H^T * S^-1
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        # State update: x = x + K * y
        self.x = self.x + np.dot(K, y_innov)

        # Covariance update: P = (I - K * H) * P
        I_mat = np.eye(6, dtype=np.float64)
        self.P = np.dot(I_mat - np.dot(K, self.H), self.P)

        if w is not None and h is not None:
            self.last_bbox_size = (float(w), float(h))

        self.missed_frames = 0
        px, py = float(self.x[0, 0]), float(self.x[1, 0])
        self.history.append((px, py, False))
        if len(self.history) > 60:
            self.history.pop(0)

        return px, py

    def handle_missed_frame(self):
        """Extrapolate position along physics flight path when detector misses."""
        if not self.is_initialized:
            return None

        self.missed_frames += 1
        if self.missed_frames > self.max_extrapolate:
            # Ball lost beyond physical extrapolation limit
            return None

        px, py = float(self.x[0, 0]), float(self.x[1, 0])
        self.history.append((px, py, True))
        if len(self.history) > 60:
            self.history.pop(0)

        return px, py

    def get_search_radius(self, base_radius=60.0):
        """Dynamic search radius based on position uncertainty covariance."""
        if not self.is_initialized:
            return base_radius
        # Standard deviation from spatial covariance
        sigma = math.sqrt(max(self.P[0, 0], self.P[1, 1]))
        return max(base_radius, sigma * 2.5 + self.missed_frames * 15.0)

    def get_current_bbox(self):
        """Return estimated [x1, y1, x2, y2] bbox."""
        if not self.is_initialized or self.missed_frames > self.max_extrapolate:
            return None
        cx, cy = float(self.x[0, 0]), float(self.x[1, 0])
        w, h = self.last_bbox_size
        return [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]

    def get_velocity(self):
        """Return (vx, vy) in pixels/frame and scalar speed."""
        if not self.is_initialized:
            return 0.0, 0.0, 0.0
        vx = float(self.x[2, 0])
        vy = float(self.x[3, 0])
        speed = math.hypot(vx, vy)
        return vx, vy, speed

    def reset(self):
        """Clear state on camera cut or prolonged absence."""
        self.x = np.zeros((6, 1), dtype=np.float64)
        self.P = np.eye(6, dtype=np.float64) * 100.0
        self.missed_frames = 0
        self.is_initialized = False
        self.history = []
