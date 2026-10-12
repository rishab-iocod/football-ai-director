"""Temporal Region-of-Interest (RoI) Dynamic Patch Cropper.

Solves the resolution bottleneck in sports ball detection:
- In full-frame 1080p, a football is only ~15-25 pixels wide.
- Downscaling 1920x1080 to standard YOLO 640x640 reduces the ball to 5-8 pixels,
  causing severe miss rates (dropping 6 out of 10 frames).
- Temporal RoI extracts a full-resolution local crop around the predicted ball location,
  enabling high-recall native-resolution inference with negligible compute latency.
"""

import numpy as np


class TemporalRoICropper:
    """Extracts dynamic native-resolution patches around predicted ball trajectory."""

    def __init__(self,
                 base_crop_size=288,
                 max_crop_size=448,
                 frame_size=(1920, 1080)):
        self.base_crop_size = base_crop_size
        self.max_crop_size = max_crop_size
        self.frame_w, self.frame_h = frame_size

    def compute_crop_box(self, predicted_center, search_radius=60.0):
        """Compute the clamped [x1, y1, x2, y2] crop window in full-frame coordinates.

        Expands crop size if search radius / uncertainty increases.
        """
        cx, cy = predicted_center

        # Adjust crop size based on trajectory uncertainty
        crop_dim = int(np.clip(search_radius * 3.0, self.base_crop_size, self.max_crop_size))
        # Ensure dimension is multiple of 32 for optimal YOLO convolutions
        crop_dim = (crop_dim // 32) * 32

        half_w = crop_dim / 2.0
        half_h = crop_dim / 2.0

        x1 = cx - half_w
        y1 = cy - half_h
        x2 = cx + half_w
        y2 = cy + half_h

        # Clamp to frame edges while maintaining crop_dim size
        if x1 < 0:
            x2 = min(self.frame_w, x2 + (0 - x1))
            x1 = 0
        if y1 < 0:
            y2 = min(self.frame_h, y2 + (0 - y1))
            y1 = 0
        if x2 > self.frame_w:
            x1 = max(0, x1 - (x2 - self.frame_w))
            x2 = self.frame_w
        if y2 > self.frame_h:
            y1 = max(0, y1 - (y2 - self.frame_h))
            y2 = self.frame_h

        return int(x1), int(y1), int(x2), int(y2)

    def extract_patch(self, frame, crop_box):
        """Extract image slice for the crop box."""
        x1, y1, x2, y2 = crop_box
        return frame[y1:y2, x1:x2].copy()

    def map_to_global_coords(self, local_bbox, crop_box):
        """Transform detection bounding box from patch coordinates back to full-frame space.

        local_bbox: [lx1, ly1, lx2, ly2]
        crop_box: [gx1, gy1, gx2, gy2]
        """
        gx1, gy1 = crop_box[0], crop_box[1]
        return [
            float(local_bbox[0] + gx1),
            float(local_bbox[1] + gy1),
            float(local_bbox[2] + gx1),
            float(local_bbox[3] + gy1),
        ]
