"""server.py — FastAPI Web Application & Streaming Server.

Hosts:
- Unified Broadcast Web Studio UI
- Low-latency MJPEG feeds: /stream/program, /stream/cam1..3, /stream/ball_focus, /stream/player_focus, /stream/roi_patch
- High-frequency WebSocket /ws for real-time telemetry, 2D radar, scores, and instant event notifications
- Interactive REST endpoints for camera switching, pan/zoom control, playback, and test triggers
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from web_platform.engine import MatchEngine

STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="Football AI Broadcast Studio",
    description="Multi-Camera Director & Tracking Platform",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = MatchEngine()


# ---------------------------------------------------------------- Models
class SourceSelectRequest(BaseModel):
    source_key: str


class CameraCutRequest(BaseModel):
    camera_id: str  # "auto", "cam1", "cam2", "cam3", "ball_focus", "player_focus"


class PTZRequest(BaseModel):
    zoom: Optional[float] = None
    pan_x: Optional[float] = None
    pan_y: Optional[float] = None
    mode: Optional[str] = None


class SpeedRequest(BaseModel):
    speed: float


class TriggerGoalRequest(BaseModel):
    team: int = 1


class TriggerFoulRequest(BaseModel):
    severity: str = "HARD_TACKLE"


# ---------------------------------------------------------------- Streaming Generator
def make_mjpeg_stream(channel):
    """Generator that yields JPEG frames as multipart/x-mixed-replace stream."""
    def stream_generator():
        seq = 0
        while True:
            seq, jpeg_bytes = channel.wait_newer(seq, timeout=0.8)
            if jpeg_bytes is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
                )
            else:
                # Keep-alive heartbeat
                yield b""

    return StreamingResponse(
        stream_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ---------------------------------------------------------------- Video Stream Routes
@app.get("/stream/program")
def stream_program():
    return make_mjpeg_stream(engine.channel_program)


@app.get("/stream/cam1")
def stream_cam1():
    return make_mjpeg_stream(engine.channel_cam1)


@app.get("/stream/cam2")
def stream_cam2():
    return make_mjpeg_stream(engine.channel_cam2)


@app.get("/stream/cam3")
def stream_cam3():
    return make_mjpeg_stream(engine.channel_cam3)


@app.get("/stream/tv")
def stream_tv():
    return make_mjpeg_stream(engine.channel_tv)


@app.get("/stream/clean")
def stream_clean():
    return make_mjpeg_stream(engine.channel_clean)


@app.get("/stream/ball_focus")
def stream_ball_focus():
    return make_mjpeg_stream(engine.channel_ball_focus)


@app.get("/stream/player_focus")
def stream_player_focus():
    return make_mjpeg_stream(engine.channel_player_focus)


@app.get("/stream/roi_patch")
def stream_roi_patch():
    return make_mjpeg_stream(engine.channel_roi_patch)


@app.get("/stream/panoramic")
@app.get("/stream/pano")
def stream_panoramic():
    return make_mjpeg_stream(engine.channel_panoramic)


# ---------------------------------------------------------------- REST Control APIs
@app.get("/api/status")
def get_status():
    return {
        "running": engine.running,
        "paused": engine.paused,
        "source": engine.current_source_key,
        "fps": round(engine.fps, 1),
        "frame_index": engine.frame_idx,
        "speed": engine.playback_speed,
        "active_cam": engine.multi_director.current_cam,
        "ptz_mode": engine.ptz_camera.mode,
        "zoom": engine.ptz_camera.current_zoom,
    }


@app.get("/api/sources")
def get_sources():
    return engine.get_sources_catalog()


@app.post("/api/source/select")
def select_source(req: SourceSelectRequest):
    res = engine.switch_source(source_key=req.source_key)
    return res


@app.post("/api/playback/start")
def start_playback():
    return engine.start()


@app.post("/api/playback/pause")
def pause_playback():
    return engine.pause()


@app.post("/api/playback/stop")
def stop_playback():
    return engine.stop()


@app.post("/api/playback/speed")
def set_playback_speed(req: SpeedRequest):
    engine.playback_speed = float(req.speed)
    return {"speed": engine.playback_speed}


@app.post("/api/director/cut")
def cut_camera(req: CameraCutRequest):
    return engine.set_camera_cut(req.camera_id)


@app.get("/api/director/config")
def get_director_config():
    return engine.multi_director.get_config()


@app.post("/api/director/config")
def set_director_config(cfg: dict):
    engine.multi_director.update_config(cfg)
    return engine.multi_director.get_config()


@app.post("/api/director/ptz")
def update_ptz(req: PTZRequest):
    return engine.set_ptz(
        zoom=req.zoom,
        pan_x=req.pan_x,
        pan_y=req.pan_y,
        mode=req.mode,
    )


@app.post("/api/trigger/goal")
def trigger_goal(req: TriggerGoalRequest):
    evt = engine.trigger_test_goal(team=req.team)
    return evt.to_dict()


@app.post("/api/trigger/foul")
def trigger_foul(req: TriggerFoulRequest):
    evt = engine.trigger_test_foul(severity=req.severity)
    return evt.to_dict()


@app.get("/api/events")
def get_events():
    return engine.events_log


# ---------------------------------------------------------------- WebSocket
@app.websocket("/ws")
async def websocket_telemetry(websocket: WebSocket):
    await websocket.accept()
    queue = asyncio.Queue(maxsize=10)
    engine.clients.add(queue)
    if engine.loop is None:
        engine.loop = asyncio.get_event_loop()

    # Send initial state
    await websocket.send_json({
        "type": "init",
        "sources": engine.get_sources_catalog(),
        "status": {
            "running": engine.running,
            "source": engine.current_source_key,
        },
        "recent_events": engine.events_log[:10],
    })

    async def sender():
        while True:
            msg = await queue.get()
            await websocket.send_json(msg)

    async def receiver():
        while True:
            data = await websocket.receive_json()
            cmd = data.get("action")
            if cmd == "cut":
                engine.set_camera_cut(data.get("camera_id", "auto"))
            elif cmd == "ptz":
                engine.set_ptz(
                    zoom=data.get("zoom"),
                    pan_x=data.get("pan_x"),
                    pan_y=data.get("pan_y"),
                    mode=data.get("mode"),
                )
            elif cmd == "playback":
                op = data.get("op")
                if op == "play":
                    engine.start()
                elif op == "pause":
                    engine.pause()
                elif op == "stop":
                    engine.stop()
            elif cmd == "source":
                engine.switch_source(data.get("source_key", "multi_cams"))
            elif cmd == "trigger_goal":
                engine.trigger_test_goal(team=data.get("team", 1))
            elif cmd == "trigger_foul":
                engine.trigger_test_foul(severity=data.get("severity", "HARD_TACKLE"))

    send_task = asyncio.create_task(sender())
    recv_task = asyncio.create_task(receiver())

    try:
        await asyncio.gather(send_task, recv_task)
    except (WebSocketDisconnect, Exception):
        pass
    finally:
        send_task.cancel()
        recv_task.cancel()
        engine.clients.discard(queue)


# ---------------------------------------------------------------- Frontend Mount
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>Broadcast Studio Frontend Not Found</h1>"


@app.get("/tv", response_class=HTMLResponse)
def tv_broadcast():
    html_path = STATIC_DIR / "tv.html"
    if html_path.exists():
        return html_path.read_text(encoding="utf-8")
    return "<h1>TV Broadcast Feed Frontend Not Found</h1>"


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def main():
    # Auto-start engine on Alfheim multi-camera match
    print("[Server] Starting MatchEngine...")
    engine.start("multi_cams")
    uvicorn.run(app, host="0.0.0.0", port=8088, log_level="info")


if __name__ == "__main__":
    main()
