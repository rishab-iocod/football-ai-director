#!/usr/bin/env python3
"""download_alfheim.py

Downloads and prepares multi-angle synchronized video footage from the
Simula Alfheim Dataset (Tromsø IL vs Strømsgodset, First Half).

The Alfheim stadium dataset features 3 synchronized cameras:
  - Camera 0: Left Wing / Goal area      -> mapped to Cam 1
  - Camera 1: Center Midfield            -> mapped to Cam 2
  - Camera 2: Right Wing / Goal area     -> mapped to Cam 3

Features:
  - Fetches synchronized 3-second .h264 video chunks
  - Downloads across cameras in parallel with automatic retry & resume
  - Concatenates and transcodes each angle to clean H.264/MP4 video (25 FPS)
  - Links output videos to data/cams/ (cam1.mp4, cam2.mp4, cam3.mp4) for
    instant compatibility with the AI Broadcast Director platform.

Usage:
  python scripts/download_alfheim.py --segments 100     # 5-minute clip (~300 MB)
  python scripts/download_alfheim.py --segments 20      # 1-minute clip (~60 MB)
  python scripts/download_alfheim.py --all              # Full 45-minute half
"""

import argparse
import os
import re
import subprocess
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import cv2
except ImportError:
    print("[ERROR] OpenCV ('cv2') is required. Please install opencv-python.")
    sys.exit(1)

# Default paths
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
ALFHEIM_DIR = DATA_DIR / "alfheim"
RAW_DIR = ALFHEIM_DIR / "raw"
CAMS_DIR = DATA_DIR / "cams"

BASE_URL = "https://datasets.simula.no/downloads/alfheim/2013-11-03/First%20Half/"


def fetch_file_list(cam_id: int) -> list[str]:
    """Retrieves list of .h264 segment filenames for a given camera."""
    url = f"{BASE_URL}{cam_id}/"
    print(f"[*] Fetching segment directory index for Camera {cam_id}...")
    try:
        html = subprocess.check_output(["curl", "-s", url], timeout=30).decode("utf-8")
    except Exception as e:
        print(f"[ERROR] Failed to fetch index for Cam {cam_id}: {e}")
        return []

    files = re.findall(r'href=[\"\']\./([^\"\']+\.h264)[\"\']', html)
    # Decode URL percent-encodings
    files = [urllib.parse.unquote(f) for f in files]
    return sorted(files)


def download_single_chunk(url: str, dst_path: Path, max_retries: int = 3) -> bool:
    """Downloads a single segment chunk via curl with retry logic."""
    if dst_path.exists() and dst_path.stat().st_size > 10000:
        return True  # Already downloaded

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dst_path.with_suffix(".tmp")

    for attempt in range(1, max_retries + 1):
        try:
            cmd = [
                "curl",
                "-s",
                "-f",
                "--retry", "2",
                "--connect-timeout", "15",
                url,
                "-o", str(temp_path),
            ]
            res = subprocess.run(cmd, capture_output=True, timeout=60)
            if res.returncode == 0 and temp_path.exists() and temp_path.stat().st_size > 1000:
                temp_path.rename(dst_path)
                return True
        except Exception:
            pass
        time.sleep(0.5 * attempt)

    if temp_path.exists():
        temp_path.unlink()
    return False


def transcode_h264_to_mp4(
    h264_path: Path,
    mp4_path: Path,
    target_res: tuple[int, int] = (960, 720),
    fps: float = 25.0,
):
    """Transcodes a concatenated .h264 file to an MP4 video using OpenCV VideoWriter."""
    print(f"[*] Transcoding {h264_path.name} -> {mp4_path.name} ({target_res[0]}x{target_res[1]} @ {fps} fps)...")
    cap = cv2.VideoCapture(str(h264_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {h264_path}")

    mp4_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(mp4_path), fourcc, fps, target_res)

    frames_written = 0
    t0 = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame.shape[1], frame.shape[0]) != target_res:
            frame = cv2.resize(frame, target_res)
        writer.write(frame)
        frames_written += 1
        if frames_written % 250 == 0:
            print(f"    ... processed {frames_written} frames ({frames_written / fps:.1f}s)", end="\r")

    cap.release()
    writer.release()
    elapsed = max(0.1, time.time() - t0)
    print(f"\n[+] Created {mp4_path.name}: {frames_written} frames ({frames_written / fps:.1f}s) in {elapsed:.1f}s ({frames_written / elapsed:.1f} FPS)")


def main():
    parser = argparse.ArgumentParser(description="Download synchronized multi-angle match footage from Simula Alfheim.")
    parser.add_argument("--segments", type=int, default=100, help="Number of 3s segments to download (default: 100 = 5 min)")
    parser.add_argument("--all", action="store_true", help="Download all available segments for the full 45m half (942 segments)")
    parser.add_argument("--workers", type=int, default=6, help="Concurrent download workers (default: 6)")
    parser.add_argument("--res", type=str, default="960x720", help="Output MP4 resolution WxH (default: 960x720)")
    parser.add_argument("--fps", type=float, default=25.0, help="Output MP4 frame rate (default: 25.0)")
    parser.add_argument("--no-transcode", action="store_true", help="Skip MP4 transcoding, download raw only")
    args = parser.parse_args()

    w_str, h_str = args.res.split("x")
    target_res = (int(w_str), int(h_str))

    print("=" * 70)
    print(" ⚽ Simula Alfheim Dataset Multi-Camera Downloader")
    print(" Match: 2013-11-03 Tromsø IL vs Strømsgodset (First Half)")
    print("=" * 70)

    # 1. Fetch file lists for all 3 cameras
    cam_lists = {}
    for c in [0, 1, 2]:
        cam_lists[c] = fetch_file_list(c)
        if not cam_lists[c]:
            print(f"[ERROR] Could not obtain file list for Cam {c}. Exiting.")
            sys.exit(1)

    # Find common synchronized segments
    common = sorted(list(set(cam_lists[0]).intersection(cam_lists[1]).intersection(cam_lists[2])))
    total_avail = len(common)
    print(f"[+] Total synchronized segments available across all 3 cameras: {total_avail} (~{total_avail * 3 / 60:.1f} min)")

    num_segs = total_avail if args.all else min(args.segments, total_avail)
    selected_segments = common[:num_segs]
    duration_s = num_segs * 3
    print(f"[*] Selected {num_segs} segments ({duration_s} seconds / {duration_s / 60:.1f} minutes of play)")
    print(f"    From: {selected_segments[0]}")
    print(f"    To:   {selected_segments[-1]}")

    # 2. Plan download tasks: download in interleaved order so partial downloads are always synchronized
    tasks = []
    for seg_name in selected_segments:
        quoted_name = urllib.parse.quote(seg_name)
        for cam_id in [0, 1, 2]:
            url = f"{BASE_URL}{cam_id}/{quoted_name}"
            dst = RAW_DIR / f"cam{cam_id}" / seg_name
            tasks.append((url, dst, cam_id, seg_name))

    total_tasks = len(tasks)
    print(f"\n[*] Starting download of {total_tasks} segment files across 3 cameras using {args.workers} threads...")

    completed = 0
    skipped = 0
    failed = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        future_map = {
            executor.submit(download_single_chunk, url, dst): (cam_id, seg_name)
            for (url, dst, cam_id, seg_name) in tasks
        }
        for future in as_completed(future_map):
            cam_id, seg_name = future_map[future]
            try:
                ok = future.result()
                if ok:
                    completed += 1
                else:
                    failed += 1
            except Exception:
                failed += 1

            pct = (completed + failed) / total_tasks * 100
            elapsed = max(0.1, time.time() - t_start)
            rate = completed / elapsed
            print(
                f"    Progress: {completed}/{total_tasks} ({pct:5.1f}%) | {rate:.1f} chunks/sec | Failed: {failed}",
                end="\r",
            )

    print(f"\n[+] Download completed in {time.time() - t_start:.1f}s. (Success: {completed}, Failed: {failed})")

    if failed > 0:
        print(f"[WARNING] {failed} chunks failed to download. Videos might have minor gaps.")

    if args.no_transcode:
        print("[*] Skipping transcode (--no-transcode specified). Done.")
        return

    # 3. Concatenate and transcode into clean MP4s
    # Alfheim Cam 0 -> Cam 1 (Left Wing)
    # Alfheim Cam 1 -> Cam 2 (Center Midfield)
    # Alfheim Cam 2 -> Cam 3 (Right Wing)
    cam_mapping = {
        0: (1, "Left Wing / Goal (Cam 0 -> Cam 1)"),
        1: (2, "Center Midfield (Cam 1 -> Cam 2)"),
        2: (3, "Right Wing / Goal (Cam 2 -> Cam 3)"),
    }

    print("\n" + "=" * 70)
    print(" 🎬 Concatenating and Transcoding to MP4 Video Containers")
    print("=" * 70)

    for cam_src, (cam_dst, desc) in cam_mapping.items():
        print(f"\n[*] Processing Angle: {desc}...")
        cam_raw_dir = RAW_DIR / f"cam{cam_src}"
        concat_h264_path = ALFHEIM_DIR / f"cam{cam_dst}_concat.h264"
        output_mp4_path = ALFHEIM_DIR / f"cam{cam_dst}.mp4"

        # Concatenate available raw h264 files in chronological order
        with open(concat_h264_path, "wb") as outfile:
            for seg_name in selected_segments:
                seg_file = cam_raw_dir / seg_name
                if seg_file.exists():
                    with open(seg_file, "rb") as infile:
                        outfile.write(infile.read())

        # Transcode to MP4
        transcode_h264_to_mp4(
            h264_path=concat_h264_path,
            mp4_path=output_mp4_path,
            target_res=target_res,
            fps=args.fps,
        )

        # Link/copy to data/cams/
        CAMS_DIR.mkdir(parents=True, exist_ok=True)
        target_link = CAMS_DIR / f"cam{cam_dst}.mp4"
        if target_link.is_symlink() or target_link.exists():
            target_link.unlink()

        try:
            target_link.symlink_to(output_mp4_path.resolve())
            print(f"[+] Symlinked {target_link.name} -> {output_mp4_path}")
        except Exception:
            # Fallback to copy if symlink is not permitted
            import shutil
            shutil.copy2(output_mp4_path, target_link)
            print(f"[+] Copied {output_mp4_path.name} -> {target_link}")

    print("\n" + "=" * 70)
    print(" ✅ Alfheim Multi-Angle Setup Complete!")
    print(f" Videos created in: {ALFHEIM_DIR}")
    print(f" Synchronized feeds linked in: {CAMS_DIR}")
    print(" Ready to run with:")
    print("   python cam_director/web_platform/server.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
