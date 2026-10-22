"""
Step 1 — Frame extraction
Extracts frames from session videos at a configurable rate, deduplicates using
perceptual hashing, caps frames per session, and writes a CSV manifest.

Usage:
    python scripts/extract_frames.py [--fps 1] [--hash-threshold 8]
                                     [--max-per-session 500]
                                     [--config configs/train_person.yaml]
                                     [--paths configs/paths.yaml]
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import cv2
import yaml

# imagehash is a lightweight library for perceptual hashing
try:
    import imagehash
    from PIL import Image
except ImportError:
    sys.exit(
        "Missing dependency. Run: pip install imagehash Pillow\n"
        "Then re-run this script."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(p: str, base: Path) -> Path:
    """Resolve relative paths against the project root (where configs/ lives)."""
    p = Path(p)
    return p if p.is_absolute() else base / p


def parse_video_filename(filename: str, pattern: str) -> dict | None:
    """Return {'session_id': ..., 'camera': ...} or None if no match."""
    m = re.search(pattern, filename)
    if m:
        return m.groupdict()
    return None


def phash_distance(img_bgr, prev_hash):
    """Return perceptual hash of img_bgr and hamming distance from prev_hash."""
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    h = imagehash.phash(pil)
    dist = (h - prev_hash) if prev_hash is not None else 999
    return h, dist


# ---------------------------------------------------------------------------
# Core extraction
# ---------------------------------------------------------------------------

def extract_video(
    video_path: Path,
    out_dir: Path,
    session_id: str,
    camera: str,
    fps_target: float,
    hash_threshold: int,
    max_frames: int,
    manifest_rows: list,
):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path.name} — skipping.")
        return 0

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, round(src_fps / fps_target))

    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    frame_idx = 0
    prev_hash = None

    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            h, dist = phash_distance(frame, prev_hash)
            if dist >= hash_threshold:
                # timestamp in video (seconds)
                ts_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                ts_sec = ts_ms / 1000.0

                stem = f"{session_id}_{camera}_{ts_sec:.3f}s"
                out_path = out_dir / f"{stem}.jpg"
                cv2.imwrite(str(out_path), frame)

                manifest_rows.append({
                    "session_id": session_id,
                    "camera": camera,
                    "timestamp_s": f"{ts_sec:.3f}",
                    "frame_path": str(out_path),
                    "video": video_path.name,
                })

                prev_hash = h
                saved += 1

        frame_idx += 1

    cap.release()
    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 1: extract frames from videos")
    parser.add_argument("--fps", type=float, default=1.0, help="Frames per second to extract")
    parser.add_argument("--hash-threshold", type=int, default=8,
                        help="Max hamming distance between consecutive kept frames (lower = stricter dedup)")
    parser.add_argument("--max-per-session", type=int, default=500,
                        help="Maximum frames to extract per session")
    parser.add_argument("--config", default="configs/train_person.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cfg = load_yaml(project_root / args.config)
    pcfg = load_yaml(project_root / args.paths)

    videos_dir = resolve_path(pcfg["videos_dir"], project_root)
    frames_dir = resolve_path(pcfg["frames_dir"], project_root)
    video_regex = pcfg["video_regex"]

    if not videos_dir.exists():
        sys.exit(f"ERROR: videos_dir does not exist: {videos_dir}\nEdit configs/paths.yaml.")

    frames_dir.mkdir(parents=True, exist_ok=True)

    # Collect video files
    video_exts = {".mp4", ".avi", ".mov", ".mkv"}
    videos = [p for p in sorted(videos_dir.rglob("*")) if p.suffix.lower() in video_exts]
    if not videos:
        sys.exit(f"No video files found in {videos_dir}")

    print(f"Found {len(videos)} video(s) in {videos_dir}")

    manifest_rows = []
    session_counts: dict[str, int] = {}

    for video_path in videos:
        info = parse_video_filename(video_path.name, video_regex)
        if info is None:
            print(f"  [SKIP] {video_path.name} — does not match video_regex pattern")
            continue

        session_id = info.get("session_id", "unknown")
        camera = info.get("camera", "unknown")

        already = session_counts.get(session_id, 0)
        remaining = args.max_per_session - already
        if remaining <= 0:
            print(f"  [SKIP] {video_path.name} — session {session_id!r} already at cap")
            continue

        out_subdir = frames_dir / session_id
        print(f"  Extracting {video_path.name}  (session={session_id}, camera={camera})", end=" ... ", flush=True)

        n = extract_video(
            video_path=video_path,
            out_dir=out_subdir,
            session_id=session_id,
            camera=camera,
            fps_target=args.fps,
            hash_threshold=args.hash_threshold,
            max_frames=remaining,
            manifest_rows=manifest_rows,
        )

        session_counts[session_id] = already + n
        print(f"{n} frames saved")

    # Write manifest CSV
    manifest_path = frames_dir / "manifest.csv"
    fieldnames = ["session_id", "camera", "timestamp_s", "frame_path", "video"]
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    # Summary
    print("\n--- Summary ---")
    print(f"Total frames extracted : {len(manifest_rows)}")
    print(f"Sessions               : {len(session_counts)}")
    for sid, cnt in sorted(session_counts.items()):
        print(f"  {sid}: {cnt} frames")
    print(f"Manifest written to    : {manifest_path}")


if __name__ == "__main__":
    main()
