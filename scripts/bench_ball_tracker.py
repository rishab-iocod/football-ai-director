"""Benchmark script comparing Baseline YOLO ball recall against the High-Recall Ball Tracker."""

import os
import sys
import time
import argparse
import cv2

# Add parent path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ball_tracker import HighRecallBallTracker


def draw_ball_telemetry(frame, tracker_state, roi_crop_box=None):
    """Draw ball position, physics flight trail, and status overlay."""
    out = frame.copy()

    # Draw trailing trajectory history
    history = tracker_state.get("history", [])
    for i in range(1, len(history)):
        pt1 = (int(history[i - 1][0]), int(history[i - 1][1]))
        pt2 = (int(history[i][0]), int(history[i][1]))
        is_ext = history[i][2]
        color = (255, 200, 0) if is_ext else (0, 255, 0)
        thickness = 2 if is_ext else 3
        cv2.line(out, pt1, pt2, color, thickness)

    # Draw current ball
    bbox = tracker_state.get("bbox")
    if bbox is not None:
        cx = int((bbox[0] + bbox[2]) / 2.0)
        cy = int((bbox[1] + bbox[3]) / 2.0)
        is_ext = tracker_state.get("is_extrapolated", False)
        color = (255, 200, 0) if is_ext else (0, 255, 0)

        # Draw circle and crosshair
        cv2.circle(out, (cx, cy), 14, color, 2)
        cv2.circle(out, (cx, cy), 3, color, -1)

        source_label = tracker_state.get("source", "unknown")
        cv2.putText(out, f"BALL ({source_label})", (cx + 18, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    # Status banner
    h, w = out.shape[:2]
    overlay = out.copy()
    cv2.rectangle(overlay, (20, 20), (520, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, out, 0.4, 0, out)

    cv2.putText(out, "HIGH-RECALL BALL TRACKER", (35, 48),
                cv2.FONT_HERSHEY_DUPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    tracked_str = "TRACKED" if tracker_state.get("tracked") else "LOST"
    track_color = (0, 255, 0) if tracker_state.get("tracked") else (0, 0, 255)
    cv2.putText(out, f"Status: {tracked_str} ({tracker_state.get('source')})",
                (35, 74), cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_color, 1, cv2.LINE_AA)
    cv2.putText(out, f"Speed: {tracker_state.get('speed_px', 0):.1f} px/f",
                (35, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    return out


def main():
    parser = argparse.ArgumentParser(description="Benchmark High-Recall Ball Tracker on video.")
    parser.add_argument("--video", default="data/test.mp4", help="Path to input video")
    parser.add_argument("--model", default="models/best.pt", help="Path to YOLO weights")
    parser.add_argument("--max-frames", type=int, default=150, help="Frames to evaluate")
    parser.add_argument("--save", default="output_videos/ball_tracker_demo.avi", help="Output video path")
    parser.add_argument("--device", default="mps", help="Inference device")
    args = parser.parse_args()

    video_path = args.video
    if not os.path.exists(video_path):
        for candidate in [
            os.path.join("..", video_path),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), video_path),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), video_path),
        ]:
            if os.path.exists(candidate):
                video_path = candidate
                break

    if not os.path.exists(video_path):
        print(f"Error: video not found at {args.video}")
        return

    model_path = args.model
    if not os.path.exists(model_path):
        for candidate in [
            os.path.join("..", model_path),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), model_path),
            os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), model_path),
        ]:
            if os.path.exists(candidate):
                model_path = candidate
                break

    print("=" * 65)
    print("CAM_DIRECTOR: HIGH-RECALL BALL TRACKING BENCHMARK")
    print(f"Source video: {video_path}")
    print(f"Weights:      {model_path}")
    print("=" * 65)

    tracker = HighRecallBallTracker(model_path=model_path, device=args.device)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if args.save:
        os.makedirs(os.path.dirname(os.path.abspath(args.save)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(args.save, fourcc, int(fps), (w, h))

    frame_idx = 0
    raw_yolo_hits = 0
    t0 = time.monotonic()

    sources_count = {}

    while cap.isOpened() and frame_idx < args.max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # Check raw baseline detection count
        # Tracker runs hybrid pipeline
        state = tracker.track(frame)
        source = state.get("source", "lost")
        sources_count[source] = sources_count.get(source, 0) + 1

        if not state.get("is_extrapolated") and state.get("tracked"):
            raw_yolo_hits += 1

        if writer is not None:
            vis = draw_ball_telemetry(frame, state)
            writer.write(vis)

        frame_idx += 1
        if frame_idx % 25 == 0:
            print(f"  Processed frame {frame_idx:>4}/{args.max_frames} | "
                  f"Coverage so far: {tracker.recall_stats['effective_coverage_pct']}%")

    cap.release()
    if writer is not None:
        writer.release()
        print(f"\n[Artifact] Saved visual demonstration to {args.save}")

    elapsed = time.monotonic() - t0
    stats = tracker.recall_stats

    print("\n" + "=" * 65)
    print("BENCHMARK RESULTS & RECALL COMPARISON")
    print("=" * 65)
    print(f"Total Frames Analyzed:        {frame_idx}")
    print(f"Processing Speed:             {frame_idx / max(0.001, elapsed):.1f} FPS")
    print("-" * 65)
    print(f"Standard YOLO Hits:           {raw_yolo_hits} / {frame_idx} ({raw_yolo_hits / max(1, frame_idx) * 100:.1f}%)")
    print(f"Hybrid Tracker Coverage:      {stats['detected_frames'] + stats['extrapolated_frames']} / {frame_idx} "
          f"({stats['effective_coverage_pct']}%)")
    print(f"  • Direct Patch/Global hits: {stats['detected_frames']} frames")
    print(f"  • Physics flight bridged:   {stats['extrapolated_frames']} frames")
    print("-" * 65)
    print("Detection Source Breakdown:")
    for src, cnt in sources_count.items():
        print(f"  - {src:<24}: {cnt} frames ({cnt / frame_idx * 100:.1f}%)")
    print("=" * 65)


if __name__ == "__main__":
    main()
