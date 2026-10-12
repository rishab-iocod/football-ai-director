# ⚽ Football AI Broadcast Director & Web Platform

> Autonomous AI Camera Director, Continuous Panoramic Tracking, and Clean Live Broadcasting Engine for Football.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)](https://opencv.org)

---

## 📖 Overview

**Football AI Director** transforms broadcast football and ultra-wide stadium video into professional television-grade productions in real-time.

Traditional camera switching across multiple cameras creates coordinate jumps, track ID loss, and false event alerts. By leveraging continuous **panoramic stadium video** (`2880×720` / `3840×960`), this platform tracks all 22 players, referees, and the ball across a **single unified pitch coordinate space**. 

An autonomous **Virtual PTZ Broadcast Camera** smoothly pans and zooms across the stadium to follow attacking play, while generating **100% clean video streams** (free of bounding boxes, lines, or debug watermarks) ready for television broadcast and OBS Studio.

---

## 🌟 Key Features

* **🎥 AI Broadcast Virtual Camera**: Dynamically crops and frames a smooth `16:9` (`1080p`) television broadcast from the wide stadium panorama, with deadband damping, velocity look-ahead, and celebration zoom.
* **🏟️ Unified Panoramic Pitch Tracking**: Continuous $x \in [0, 2880]$ coordinate tracking eliminates multi-camera seam artifacts, ball coordinate jumps, and track resets.
* **✨ Clean Live Video Feeds**: Zero bounding boxes, `#ID` labels, trajectory lines, or debug watermarks in the live stream outputs.
* **🥅 Accurate Event Detectors**:
  * **Goals**: Pinned to the actual physical stadium goal mouths ($x < 190\text{px}$ and $x > 2690\text{px}$) with inward trajectory validation, completely eliminating false midfield goals.
  * **Fouls & Tackles**: Evaluates player-player collision proximity and physical ground tumbles in unified pitch coordinates without edge-cut false alarms.
* **🎛️ Web Studio & Multiview Switcher Desk**: Real-time browser control room with Program Monitor, 6-camera multiview wall, interactive switcher bus, PTZ controls, and live match radar.
* **📺 Dedicated Television Monitor (`/tv`)**: Fullscreen broadcast monitor with TV scoreboard bug, lower-third alert toasts, and instant toggle between TV Program and 100% Clean OBS Feed.

---

## 🚀 Quickstart: Running the Web Platform

### 1. Installation

Ensure you have Python 3.9+ installed, then install dependencies:

```bash
# Clone the repository
git clone https://github.com/rishab-iocod/football-ai-director.git
cd football-ai-director

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install fastapi uvicorn websockets opencv-python ultralytics supervision numpy scipy
```

### 2. Start the Broadcast Studio Server

Run the web platform server from the repository root:

```bash
python cam_director/web_platform/server.py
```

*The server will start at `http://0.0.0.0:8088` and automatically initialize on the default panoramic stadium feed.*

### 3. Open in Your Browser

| Interface | URL | Description |
|---|---|---|
| **Web Studio Control Room** | [http://localhost:8088](http://localhost:8088) | Multi-camera switcher desk, PTZ controls, 2D radar, telemetry |
| **Fullscreen TV / OBS Monitor** | [http://localhost:8088/tv](http://localhost:8088/tv) | Fullscreen broadcast output with scoreboard bug & OBS URL copy |

---

## 📡 Live Stream Endpoints (MJPEG)

Every stream output is available as a standard MJPEG endpoint compatible with **OBS Studio**, **VLC**, and web browsers:

| Stream Endpoint | Aspect / Res | Description |
|---|---|---|
| `http://localhost:8088/stream/tv` | 16:9 (`1920×1080`) | Main Program Broadcast feed with top-left TV scoreboard bug |
| `http://localhost:8088/stream/clean` | 16:9 (`1920×1080`) | **100% Clean Feed** — pristine raw video with zero graphics (for external graphics engines / OBS) |
| `http://localhost:8088/stream/panoramic` | 4:1 (`1920×480`) | Full Stadium Overview — continuous raw panoramic view of the entire pitch |
| `http://localhost:8088/stream/ball_focus` | 16:9 (`1280×720`) | Dynamic camera locked onto the ball |
| `http://localhost:8088/stream/player_focus` | 16:9 (`1280×720`) | Dynamic camera tracking the active hero player / ball carrier |
| `http://localhost:8088/stream/cam1` | 4:3 (`960×720`) | Raw Left Wing / Penalty Box sector |
| `http://localhost:8088/stream/cam2` | 4:3 (`960×720`) | Raw Center Midfield sector |
| `http://localhost:8088/stream/cam3` | 4:3 (`960×720`) | Raw Right Wing / Penalty Box sector |
| `http://localhost:8088/stream/roi_patch` | 1:1 (`240×240`) | High-resolution Temporal RoI patch around the ball |

### Adding to OBS Studio
1. In OBS Studio, add a **Media Source** or **Browser Source**.
2. Set the URL to `http://localhost:8088/stream/tv` (with scoreboard) or `http://localhost:8088/stream/clean` (for custom lower-thirds).
3. The stream will play with low latency at native 25 FPS.

---

## ⌨️ Operator Controls & Keyboard Hotkeys

### In the Web Studio (`/`)
* **`A`**: AI Auto Director (Autonomous virtual camera tracking)
* **`0` or `W`**: Cut to Full Stadium Panorama
* **`1`**: Cut to Left Wing (Cam 1)
* **`2`**: Cut to Center Midfield (Cam 2)
* **`3`**: Cut to Right Wing (Cam 3)
* **`B`**: Lock onto Ball Focus
* **`P`**: Lock onto Hero Player Focus
* **`Space`**: Pause / Resume match playback

### In the TV Broadcast Monitor (`/tv`)
* **`F`**: Toggle Fullscreen
* **`C`**: Instant toggle between TV Program Feed (with scoreboard) and Clean Feed (no graphics)
* **`1`, `2`, `3`**: Cut to Cam 1, Cam 2, Cam 3
* **`B`**: Ball Lock
* **`P`**: Hero Player Lock
* **`A`**: AI Auto Director
* **`H`**: Hide / Show floating toolbar

---

## ⚙️ Architecture & Data Flow

```
Raw Stadium Panorama (2880×720 @ 25 FPS)
  │
  ├── Unified Pitch Tracking (Ball & 22 Players in x ∈ [0, 2880])
  │     ├── Continuous Kalman Ball Filter (Index 0)
  │     └── Team Possession Assignment
  │
  ├── Real-Time Event Detection
  │     ├── GoalDetector (Left: x < 190, Right: x > 2690 + Inward Velocity)
  │     └── FoulDetector (Player-to-Player Proximity + Tumble Kinematics)
  │
  ├── Virtual PTZ Broadcast Camera
  │     ├── Smooth EMA & Lead Look-Ahead Kinematics
  │     ├── 16:9 Aspect Cropping (1280×720 window inside 2880×720)
  │     └── Resizing to 1920×1080 (Zero Distortion)
  │
  └── Multi-Channel Frame Dispatch
        ├── /stream/tv       (1080p + TV Scoreboard Bug)
        ├── /stream/clean    (1080p 100% Clean)
        ├── /stream/panoramic(Full Stadium Pitch)
        └── /stream/ball_focus & /stream/player_focus
```

---

## 🔬 Running CLI Tools & Benchmarks

### 1. High-Recall Ball Tracker Interactive Demo
Follow the ball live on screen with a continuous physics trail, velocity vectors, and high-resolution RoI zoom window:

```bash
# Test on a recorded video
python cam_director/scripts/live_tracker.py --source data/test.mp4 --device mps

# Test on live 24/7 sports broadcast stream (HLS)
python cam_director/scripts/live_tracker.py --source aspor --device mps
```

### 2. Ball Tracker Benchmark
Evaluate ball detection recall, physics Kalman bridging, and latency metrics:

```bash
python cam_director/scripts/bench_ball_tracker.py \
    --video data/test.mp4 \
    --model models/best.pt \
    --device mps \
    --max-frames 300 \
    --save output_videos/ball_tracker_demo.avi
```

### 3. Offline Analysis Batch Pipeline
Run full batch YOLO detection, ByteTrack, camera motion estimation, field perspective homography, and speed/distance calculation:

```bash
python main.py
```
*Output will be saved to `output_videos/output_video.avi`.*

---

## 📄 License
This project is licensed under the MIT License.
