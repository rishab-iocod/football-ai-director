"""Real-Time Interactive Live Video Ball Tracker & Visualizer.

Pulls live frames from HLS (.m3u8), RTSP, webcam, or video files and visualizes:
1. High-Recall Ball Tracking with continuous physical flight trail
2. Real-time Native-Resolution Temporal RoI patch zoom (in-picture PiP)
3. Dynamic velocity vectors and kinematic metrics
4. Interactive controls:
   - 'q' or ESC: exit
   - 'p': toggle RoI patch zoom picture-in-picture
   - 's': save instant snapshot
"""

import os
import sys
import time
import argparse
import cv2

# Add paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "cam_director"))

from live.capture import ResilientCapture
from ball_tracker import HighRecallBallTracker


LIVE_PRESETS = {
    "aspor": "https://rnttwmjcin.turknet.ercdn.net/lcpmvefbyo/aspor/aspor.m3u8",
    "rtve": "https://rtvelivestream.rtve.es/rtvesec/la1/la1_main_dvr.m3u8",
}


def draw_live_tracking_overlay(frame, tracker_state, show_patch=True, patch_img=None):
    """Draw ball position, trajectory trail, velocity arrow, and HUD banner."""
    out = frame.copy()
    h, w = out.shape[:2]

    # Draw trailing trajectory history
    history = tracker_state.get("history", [])
    for i in range(1, len(history)):
        pt1 = (int(history[i - 1][0]), int(history[i - 1][1]))
        pt2 = (int(history[i][0]), int(history[i][1]))
        is_ext = history[i][2]
        color = (0, 200, 255) if is_ext else (0, 255, 0)
        thickness = 2 if is_ext else 3
        cv2.line(out, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Draw current ball
    bbox = tracker_state.get("bbox")
    if bbox is not None:
        cx = int((bbox[0] + bbox[2]) / 2.0)
        cy = int((bbox[1] + bbox[3]) / 2.0)
        is_ext = tracker_state.get("is_extrapolated", False)
        color = (0, 200, 255) if is_ext else (0, 255, 0)

        # Circle marker
        cv2.circle(out, (cx, cy), 16, color, 2, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 4, color, -1, cv2.LINE_AA)

        # Velocity direction arrow
        vx, vy = tracker_state.get("velocity", (0.0, 0.0))
        if abs(vx) > 1.0 or abs(vy) > 1.0:
            arrow_end = (int(cx + vx * 2.5), int(cy + vy * 2.5))
            cv2.arrowedLine(out, (cx, cy), arrow_end, (0, 255, 255), 2, tipLength=0.3)

        source_label = tracker_state.get("source", "unknown").upper()
        cv2.putText(out, f"BALL [{source_label}]", (cx + 20, cy + 6),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1, cv2.LINE_AA)

    # Optional Native-Resolution Patch Zoom Picture-in-Picture
    if show_patch and patch_img is not None and patch_img.size > 0:
        pw, ph = 220, 220
        pip_patch = cv2.resize(patch_img, (pw, ph))
        # Draw target reticle in center of patch
        cv2.drawMarker(pip_patch, (pw // 2, ph // 2), (0, 255, 255),
                       cv2.MARKER_CROSS, 20, 1)
        cv2.putText(pip_patch, "TEMPORAL RoI ZOOM", (8, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(pip_patch, (0, 0), (pw - 1, ph - 1), (0, 200, 255), 2)

        # Overlay on top right
        pad = 20
        out[pad:pad + ph, w - pw - pad:w - pad] = pip_patch

    # Telemetry HUD banner (top-left)
    overlay = out.copy()
    banner_w = 460
    banner_h = 100
    cv2.rectangle(overlay, (20, 20), (20 + banner_w, 20 + banner_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, out, 0.30, 0, out)

    tracked = tracker_state.get("tracked", False)
    status_str = "TRACKED" if tracked else "SEARCHING"
    status_col = (0, 255, 0) if tracked else (0, 140, 255)
    speed = tracker_state.get("speed_px", 0.0)

    cv2.putText(out, "CAM_DIRECTOR: LIVE BALL TRACKER", (35, 46),
                cv2.FONT_HERSHEY_DUPLEX, 0.60, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"STATUS: {status_str} ({tracker_state.get('source')})", (35, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, status_col, 1, cv2.LINE_AA)
    cv2.putText(out, f"SPEED: {speed:.1f} px/f   CONF: {tracker_state.get('confidence', 0):.2f}", (35, 96),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    return out


def main():
    parser = argparse.ArgumentParser(description="Live football ball tracking with interactive visual preview.")
    parser.add_argument("--source", default="aspor",
                        help="Live HLS stream URL, webcam index (0), MP4 file path, or preset ('aspor', 'rtve')")
    parser.add_argument("--model", default="models/best.pt", help="YOLO model path")
    parser.add_argument("--device", default="mps", help="Inference device (mps/cpu/cuda)")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames")
    parser.add_argument("--save", default=None, help="Optional path to record annotated video")
    parser.add_argument("--no-preview", action="store_true", help="Run headless without OpenCV window")
    args = parser.parse_args()

    stream_source = LIVE_PRESETS.get(args.source, args.source)
    model_path = args.model
    if not os.path.exists(model_path):
        for candidate in [
            os.path.join(ROOT_DIR, model_path),
            os.path.join(ROOT_DIR, "models", "best.pt"),
            os.path.join(ROOT_DIR, "football_analysis_yolo", "models", "best.pt"),
        ]:
            if os.path.exists(candidate):
                model_path = candidate
                break

    if not stream_source.startswith("http") and not stream_source.isdigit():
        if not os.path.exists(stream_source):
            for candidate in [
                os.path.join(ROOT_DIR, stream_source),
                os.path.join(ROOT_DIR, "football_analysis_yolo", stream_source),
            ]:
                if os.path.exists(candidate):
                    stream_source = candidate
                    break

    print("=" * 65)
    print("CAM_DIRECTOR: LIVE VIDEO BALL TRACKING")
    print(f"Source:       {args.source} -> {stream_source}")
    print(f"Model:        {model_path}")
    print(f"Device:       {args.device}")
    print("Hotkeys:      [q]=quit  [p]=toggle zoom PiP  [s]=save snapshot")
    print("=" * 65)

    tracker = HighRecallBallTracker(model_path=model_path, device=args.device)
    capture = ResilientCapture(stream_source).start()

    writer = None
    show_patch_pip = True
    processed = 0
    t0 = time.monotonic()

    try:
        while True:
            if args.max_frames is not None and processed >= args.max_frames:
                break

            frame = capture.read(timeout=5.0)
            if frame is None:
                continue

            # Resize to calibration 1920x1080 if needed
            if (frame.shape[1], frame.shape[0]) != (1920, 1080):
                frame = cv2.resize(frame, (1920, 1080))

            # Track frame
            state = tracker.track(frame)

            # Get patch image if predicted
            patch_img = None
            if tracker.kalman.is_initialized:
                pred = (tracker.kalman.x[0, 0], tracker.kalman.x[1, 0])
                crop_box = tracker.cropper.compute_crop_box(pred)
                patch_img = tracker.cropper.extract_patch(frame, crop_box)

            annotated = draw_live_tracking_overlay(frame, state, show_patch=show_patch_pip, patch_img=patch_img)

            if args.save:
                if writer is None:
                    os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*"XVID")
                    writer = cv2.VideoWriter(args.save, fourcc, 24, (1920, 1080))
                writer.write(annotated)

            processed += 1
            elapsed = time.monotonic() - t0
            fps = processed / max(0.001, elapsed)

            if not args.no_preview:
                cv2.imshow("CamDirector Live Ball Tracker ('q'=quit, 'p'=toggle zoom)", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                elif key in (ord("p"), ord("P")):
                    show_patch_pip = not show_patch_pip
                elif key in (ord("s"), ord("S")):
                    snap_name = f"snapshot_frame_{processed}.jpg"
                    cv2.imwrite(snap_name, annotated)
                    print(f"[Snapshot] Saved {snap_name}")

            if processed % 30 == 0:
                print(f"  Frame {processed:>5} | FPS: {fps:.1f} | "
                      f"Status: {state.get('source')} | "
                      f"Coverage: {tracker.recall_stats['effective_coverage_pct']}%")

    except KeyboardInterrupt:
        print("\nStopping live tracker...")
    finally:
        capture.stop()
        if writer is not None:
            writer.release()
            print(f"[Saved] Output recorded to {args.save}")
        if not args.no_preview:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
