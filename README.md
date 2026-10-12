# CamDirector — AI Match Storytelling & Telemetry Engine

An intelligent multi-camera broadcast director and video telemetry pipeline for live football.

---

## The Core Problem

Following the story of a football match the way a human director does has no off-the-shelf solution. 
The system depends on generating a **real-time description of every camera feed (~1 Hz)**:
1. **Shot classification**: Tactical wide, medium play, tight close-up, crowd, bench/coach, replay.
2. **High-recall ball tracking**: Overcoming the 4/10 frame detection drop in standard football models.
3. **Play kinematics & tempo**: Speed of play, pressing intensity, counter-attack velocity.
4. **Entity awareness**: Ball carrier, sprint leaders, referees, team formations.
5. **Streaming event detection**: Real-time goals, fouls/tackles, shots on target, and celebrations.

With structured per-camera telemetry, choosing the best camera for narrative impact becomes deterministic and straightforward.

---

## 1. High-Recall Ball Tracker (`ball_tracker/`)

Open models miss the football in ~6 out of 10 frames due to extreme downscaling (a 18px ball downscales to 4px in standard 640x640 YOLO), motion blur, and leg occlusions.

`HighRecallBallTracker` solves this using a 3-layer hybrid architecture:

| Component | Responsibility | Impact |
|---|---|---|
| `TemporalRoICropper` | Extracts a $288\text{px}$ native-resolution patch around predicted trajectory | Preserves optical ball details without downscaling |
| `TrajectoryValidator` | Kinematic acceleration & aspect ratio gating | Filters false positives on white socks, sponsor logos, grass marks |
| `PhysicsBallKalman` | 6D physical Kalman filter with image-space gravity & aerodynamic drag | Bridges 1–12 frame gaps along physical parabolic flight arcs |

### Benchmark Results on Broadcast Footage (`data/test.mp4`)

```
=================================================================
BENCHMARK RESULTS & RECALL COMPARISON
=================================================================
Total Frames Analyzed:        80
-----------------------------------------------------------------
Standard YOLO Hits:           28 / 80 (35.0% recall)
Hybrid Tracker Coverage:      73 / 80 (91.2% continuous coverage)
  • Direct Patch/Global hits: 28 frames
  • Physics flight bridged:   45 frames
-----------------------------------------------------------------
Detection Source Breakdown:
  - lost                    : 7 frames (8.8%)
  - global_frame            : 5 frames (6.2%)
  - temporal_roi_patch      : 23 frames (28.7%)
  - physics_extrapolated    : 45 frames (56.2%)
=================================================================
```

---

## 2. Real-Time Camera Feed Telemetry Schema (`descriptor/`)

Defines [`CameraFeedTelemetry`](cam_director/descriptor/schema.py), the standardized ~1 Hz JSON descriptor emitted per camera feed:

```json
{
  "camera_id": "cam_01_high_center",
  "timestamp": 1728472910.42,
  "frame_index": 340,
  "shot_type": "WIDE_TACTICAL",
  "shot_confidence": 0.94,
  "camera_motion": "SMOOTH_PAN",
  "ball_visible": true,
  "ball_coords_norm": [0.54, 0.61],
  "ball_speed_px_f": 18.4,
  "ball_trajectory_status": "AIRBORNE_SHOT",
  "play_tempo": "HIGH_COUNTER",
  "attacking_speed_kmh": 26.8,
  "player_count": 14,
  "ball_carrier_id": "10",
  "is_referee_in_focus": false,
  "team_in_possession": 1,
  "event_type": "SHOT_ON_TARGET",
  "event_confidence": 0.88
}
```

---

## Quickstart

### Option A: Watch Real-Time Live Video Tracking (Interactive Window)

Watch the tracking engine follow the ball live on screen with a continuous physics trail, velocity arrows, and a high-resolution RoI zoom Picture-in-Picture window:

```bash
# 1. Real-time 24/7 sports broadcast feed
python cam_director/scripts/live_tracker.py --source aspor --device mps

# 2. Or using public broadcast feed (RTVE)
python cam_director/scripts/live_tracker.py --source rtve --device mps

# 3. Or using any custom HLS (.m3u8), RTSP, or webcam stream
python cam_director/scripts/live_tracker.py --source "https://YOUR_STREAM_URL.m3u8" --device mps
```

> **Interactive Hotkeys in Preview Window:**
> * Press **`q`** or **`ESC`**: Quit
> * Press **`p`**: Toggle the high-resolution Temporal RoI Zoom PiP on/off
> * Press **`s`**: Save an instant full-resolution snapshot

---

### Option B: Test with the Longer Recorded Match Video (`data/match_sample_long.mp4`)

We captured a new, longer 1080p match video (`40 MB`, 600 frames) in `data/match_sample_long.mp4`:

```bash
# Run interactive visual tracker on the long match recording
python cam_director/scripts/live_tracker.py --source data/match_sample_long.mp4 --device mps

# Or run the benchmark metrics and save annotated output
python cam_director/scripts/bench_ball_tracker.py \
    --video data/match_sample_long.mp4 \
    --model models/best.pt \
    --device mps \
    --max-frames 300 \
    --save output_videos/ball_tracker_long_demo.avi
```

