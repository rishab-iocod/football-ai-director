# ⚽ Football AI Broadcast Director & Web Platform

> Autonomous AI Multi-Camera Director, Synchronized Stadium Tracking, Virtual PTZ Camera, and Clean Live Broadcasting Engine for Football.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-red.svg)](https://opencv.org)

---

## 📖 Overview

**Football AI Director** transforms synchronized multi-angle stadium cameras (e.g. Simula Alfheim Dataset) and ultra-wide panoramic stadium video into professional television-grade productions in real-time.

Traditional camera switching across broadcast cameras suffers from coordinate desynchronization, tracking ID dropouts, zoom breathing, and false event alarms. This platform introduces:

1. **Autonomous Multi-Camera Director**: Decides between **Left Wing (Cam 1)**, **Center Midfield (Cam 2)**, and **Right Wing (Cam 3)** in real-time based on ball possession, carrier kinematics, and field dominance.
2. **Asynchronous Inference Pipeline**: Runs deep YOLOv8 detection in a background worker while video streams run at a rock-solid, fluid **25.0 FPS wall-clock time** with zero playback lag.
3. **Stabilized Virtual PTZ Camera**: Smoothly tracks the ball and players with horizon-clamped vertical deadbands and zoom-hold stabilization (eliminating camera bobbing and zoom pulsing).
4. **Unified 2D Pitch Radar & Tactical Minimap**: Stitches all 3 camera perspectives into global field coordinates $[0.0, 1.0]$, deduplicating players in camera overlap zones so all 22 players are continuously visible without teleporting during camera cuts.
5. **Geometry-Aware Match Event Detectors**: Physics-based Goal and Foul engines verified against real pitch coordinates and multi-frame collision persistence.

---

## 🌟 Key Features

* **🎥 AI Broadcast Virtual Camera**: Smoothly frames a true `16:9` (`1080p`) broadcast from the active camera angle, with deadband damping, velocity look-ahead, and celebration close-ups.
* **⚡ Decoupled 25 FPS Video Engine**: Main video playback is completely decoupled from YOLO inference via thread-safe frame queues. Playback remains locked at 25.0 FPS with zero buffering or stuttering.
* **🏟️ Multi-Camera Angle Switching (Cam 1, Cam 2, Cam 3)**:
  * Automatically cuts to the camera angle covering the active play with agile hysteresis (~0.28s) and minimum shot stability (~1.0s).
  * Direct cut re-centering snaps the virtual PTZ focus directly onto the action in the new angle without jarring pan sweeps.
  * Explicit static pitch-marking rejection ignores decoy turf spots (e.g. penalty spots and turf circles).
* **🗺️ Continuous Full-Pitch 2D Radar**:
  * Projects **Cam 1** ($0.0 \to 0.36$), **Cam 2** ($0.32 \to 0.68$, center circle at $0.50$), and **Cam 3** ($0.64 \to 1.00$) into a unified bird's-eye view.
  * Spatial deduplication in the camera overlap zones prevents duplicate player blips.
  * Live gliding ball trajectory tracks across the entire pitch seamlessly.
* **🥅 Physical Event Detectors**:
  * **Goals**: Camera-aware goal mouth enforcement. Midfield (Cam 2) is physically barred from triggering goals; goals only trigger at Left Net ($x < 140\text{px}, v_x < -1.8$) in Cam 1 or Right Net ($x > 820\text{px}, v_x > +1.8$) in Cam 3.
  * **Fouls & Tackles**: Temporal persistence gate ($\ge 4$ consecutive low frames) rejects single-frame bounding box dropouts, with strict physical contact ($\le 48\text{px}$) and ball contest proximity ($\le 130\text{px}$).
* **✨ 100% Clean Live Video Feeds**: Zero bounding boxes, debug labels, or watermarks in broadcast outputs—ready for television broadcast and OBS Studio.
* **🎛️ Interactive Web Studio (`/`)**: Multi-camera switcher desk, Program Monitor, 6-camera multiview wall, PTZ pan/zoom controls, and live match telemetry.
* **📺 Fullscreen Television Monitor (`/tv`)**: Production monitor with TV scoreboard bug, alert toasts, and instant toggle between TV Program and Clean OBS feed.

---

## 🚀 Quickstart

### 1. Installation

Ensure you have Python 3.9+ installed:

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

### 2. Download Synchronized Multi-Angle Footage (Simula Alfheim)

The platform includes a parallel downloader for the official **Simula Alfheim Dataset** (Tromsø IL vs Strømsgodset):

```bash
# Download 100 synchronized segments (~5 minutes of match footage across 3 cameras)
python cam_director/scripts/download_alfheim.py --segments 100
```
*This downloads synchronized `.h264` chunks for Cam 0 (Left), Cam 1 (Midfield), and Cam 2 (Right), transcodes them to clean 25 FPS MP4s, and places them into `data/cams/` (`cam1.mp4`, `cam2.mp4`, `cam3.mp4`).*

### 3. Launch the Broadcast Studio Server

```bash
python cam_director/web_platform/server.py
```

*The server will start at `http://0.0.0.0:8088`.*

### 4. Open in Your Browser

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
| `http://localhost:8088/stream/clean` | 16:9 (`1920×1080`) | **100% Clean Feed** — pristine raw video with zero graphics (for OBS / external engines) |
| `http://localhost:8088/stream/cam1` | 4:3 (`960×720`) | Raw Left Wing / Penalty Box angle |
| `http://localhost:8088/stream/cam2` | 4:3 (`960×720`) | Raw Center Midfield angle |
| `http://localhost:8088/stream/cam3` | 4:3 (`960×720`) | Raw Right Wing / Penalty Box angle |
| `http://localhost:8088/stream/panoramic` | 4:1 (`1920×480`) | Full Stadium Overview — continuous panoramic pitch |
| `http://localhost:8088/stream/ball_focus` | 16:9 (`1280×720`) | Dedicated dynamic crop locked on the ball |
| `http://localhost:8088/stream/player_focus` | 16:9 (`1280×720`) | Dedicated dynamic crop tracking the active ball carrier |
| `http://localhost:8088/stream/roi_patch` | 1:1 (`240×240`) | High-resolution Temporal RoI patch around the ball |

### Adding to OBS Studio
1. In OBS Studio, add a **Browser Source** or **Media Source**.
2. Set the URL to `http://localhost:8088/stream/tv` (with scoreboard) or `http://localhost:8088/stream/clean` (clean feed).
3. Set the width to `1920` and height to `1080`.

---

## ⌨️ Operator Controls & Keyboard Hotkeys

### In the Web Studio (`/`)
* **`A`**: AI Auto Director (Autonomous camera switching & PTZ tracking)
* **`1`**: Force cut to Left Wing (Cam 1)
* **`2`**: Force cut to Center Midfield (Cam 2)
* **`3`**: Force cut to Right Wing (Cam 3)
* **`0` or `W`**: Cut to Full Stadium Panorama
* **`B`**: Lock onto Ball Focus feed
* **`P`**: Lock onto Hero Player Focus feed
* **`Space`**: Pause / Resume match playback

### In the TV Broadcast Monitor (`/tv`)
* **`F`**: Toggle Fullscreen
* **`C`**: Instant toggle between TV Program Feed (with scoreboard) and Clean Feed (no graphics)
* **`1`, `2`, `3`**: Cut to Cam 1, Cam 2, Cam 3
* **`A`**: Re-engage Autonomous AI Director
* **`H`**: Hide / Show floating toolbar

---

## ⚙️ Architecture & Data Flow

```
Synchronized Multi-Cameras (Cam 1 Left, Cam 2 Mid, Cam 3 Right @ 25 FPS)
  │
  ├── Asynchronous Inference Worker (YOLOv8 on 480×360 in background thread)
  │     ├── Equal Round-Robin Scheduling across cameras (~240ms cycle)
  │     └── Static Turf Mark & Decoy Rejection
  │
  ├── Autonomous Multi-Camera Director
  │     ├── Ball Dominance & Possession Scoring (Ball = +100..+130, Players ≤ 12)
  │     ├── Hysteresis (~0.28s) & Minimum Shot Stability (~1.0s)
  │     └── Broadcast Cut Dispatch
  │
  ├── Virtual PTZ Broadcast Camera
  │     ├── Direct Cut Re-Centering (no sweeping across stadium)
  │     ├── Horizon-Clamped Vertical Deadband (eliminates bobbing)
  │     ├── Zoom Deadband (stops breathing/pulsing, holds 1.25× framing)
  │     └── Resizing to 1920×1080 16:9 Full HD
  │
  ├── Full-Pitch 2D Radar Fusion
  │     ├── Sector Projection: Cam 1 [0..0.36], Cam 2 [0.32..0.68], Cam 3 [0.64..1.0]
  │     ├── Boundary Overlap Deduplication
  │     └── Live Gliding Ball Pitch Projection
  │
  ├── Physics-Based Event Detectors
  │     ├── GoalDetector: Net mouth verification (Cam 1 Left / Cam 3 Right only)
  │     └── FoulDetector: Temporal persistence (≥ 4 low frames) + physical collision
  │
  └── Multi-Channel Frame & Telemetry Dispatch
        ├── MJPEG Streaming Channels (/stream/tv, /stream/clean, etc.)
        └── High-Frequency WebSocket (/ws) for Radar & Live Telemetry
```

---

## 📄 License
This project is licensed under the MIT License.
