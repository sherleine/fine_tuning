"""
Simple detection script — runs YOLOE on an image or video,
draws bounding boxes + confidence scores, and saves results.

Usage:
    python scripts/detect.py --source path/to/video.mp4
    python scripts/detect.py --source path/to/photo.jpg
    python scripts/detect.py --source path/to/video.mp4 --model finetuned
    python scripts/detect.py --source path/to/video.mp4 --every 3

Output folder:  runs/detections/<source_stem>/
    annotated.mp4 / annotated.jpg   — video/image with boxes drawn
    scores.csv                      — frame, timestamp, count, confidence scores
    summary.txt                     — overall stats

--model baseline   uses stock YOLOE-26s (zero-shot)   [default]
--model finetuned  uses the fine-tuned checkpoint
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

# Box and text colours per model
COLORS = {
    "baseline":  (0, 220, 80),    # green
    "finetuned": (30, 140, 255),  # blue-orange
}


def find_finetuned_weights(run_name="yoloe_finetune") -> Path | None:
    candidates = sorted(Path("runs").glob(f"**/{run_name}*/weights/best.pt"))
    if not candidates:
        candidates = sorted(Path("runs").glob("**/weights/best.pt"))
    return candidates[-1] if candidates else None


def load_model(model_type: str):
    from ultralytics import YOLOE
    if model_type == "baseline":
        print("Loading stock YOLOE-26s (zero-shot) ...")
        m = YOLOE("yoloe-26s-seg.pt")
        m.set_classes(["person"])
        return m
    elif model_type == "finetuned":
        weights = find_finetuned_weights()
        if weights is None:
            sys.exit("No fine-tuned weights found. Run finetune_compare.py first.")
        print(f"Loading fine-tuned model: {weights}")
        return YOLOE(str(weights))
    else:
        sys.exit(f"--model must be 'baseline' or 'finetuned', got: {model_type}")


def annotate_frame(frame: np.ndarray, result, color, model_label: str,
                   frame_idx: int, ts: float) -> tuple[np.ndarray, list[float]]:
    img   = frame.copy()
    h, w  = img.shape[:2]
    confs = []

    if result.boxes is not None and len(result.boxes) > 0:
        for box, c in zip(result.boxes.xyxy.cpu().numpy(),
                          result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            confs.append(float(c))

            # Bounding box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

            # Confidence label with filled background
            text = f"person  {c:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            label_y = max(y1 - 6, th + 6)
            cv2.rectangle(img,
                          (x1, label_y - th - 4),
                          (x1 + tw + 8, label_y + 2),
                          color, cv2.FILLED)
            cv2.putText(img, text, (x1 + 4, label_y - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                        cv2.LINE_AA)

    count    = len(confs)
    avg_conf = sum(confs) / count if confs else 0.0
    max_conf = max(confs)         if confs else 0.0

    # Info strip at the top
    strip_h = 48
    strip   = np.zeros((strip_h, w, 3), dtype=np.uint8)
    strip[:] = (25, 25, 25)

    # Left: model name + frame info
    cv2.putText(strip, f"{model_label}", (10, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1, cv2.LINE_AA)
    cv2.putText(strip, f"frame {frame_idx}   t = {ts:.2f}s", (10, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Right: detection stats
    stats = (f"detected: {count}    avg conf: {avg_conf:.3f}    max conf: {max_conf:.3f}"
             if count else "detected: 0")
    (sw, _), _ = cv2.getTextSize(stats, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
    cv2.putText(strip, stats, (w - sw - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 1, cv2.LINE_AA)

    return np.vstack([strip, img]), confs


def process_image(source: Path, model, color: str, model_label: str, out_dir: Path):
    img = cv2.imread(str(source))
    if img is None:
        sys.exit(f"Cannot read image: {source}")

    from ultralytics import YOLOE
    result = model.predict(source=img, conf=0.25, verbose=False)[0]
    annotated, confs = annotate_frame(img, result, color, model_label, 0, 0.0)

    out_img = out_dir / "annotated.jpg"
    cv2.imwrite(str(out_img), annotated)

    count    = len(confs)
    avg_conf = sum(confs) / count if confs else 0.0

    # Summary
    print()
    print("=" * 48)
    print(f"  Detections   : {count}")
    if confs:
        print(f"  Confidences  : {', '.join(f'{c:.4f}' for c in confs)}")
        print(f"  Avg conf     : {avg_conf:.4f}")
        print(f"  Max conf     : {max(confs):.4f}")
    print("=" * 48)
    print(f"\nSaved → {out_img}")


def process_video(source: Path, model, color, model_label: str,
                  out_dir: Path, every: int):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {source}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = out_dir / "annotated.mp4"
    csv_path = out_dir / "scores.csv"
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(str(out_path), fourcc,
                               fps / every, (w_orig, h_orig + 48))

    print(f"Video     : {source.name}")
    print(f"Size      : {w_orig}x{h_orig} @ {fps:.1f} fps  |  {total} frames")
    print(f"Sampling  : every {every} frame(s)")
    print(f"Output    : {out_path}")
    print(f"CSV       : {csv_path}")
    print()

    csv_rows    = []
    all_confs   = []
    frame_counts = []
    frame_idx   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every == 0:
            ts = frame_idx / fps
            result = model.predict(source=frame, conf=0.25, verbose=False)[0]
            annotated, confs = annotate_frame(
                frame, result, color, model_label, frame_idx, ts)

            writer.write(annotated)
            all_confs.extend(confs)
            frame_counts.append(len(confs))

            csv_rows.append({
                "frame":    frame_idx,
                "time_s":   f"{ts:.3f}",
                "count":    len(confs),
                "avg_conf": f"{sum(confs)/len(confs):.4f}" if confs else "0",
                "max_conf": f"{max(confs):.4f}" if confs else "0",
                "all_confs": ";".join(f"{c:.4f}" for c in confs),
            })

            if len(csv_rows) % 30 == 1:
                pct = 100 * frame_idx / max(total, 1)
                avg = sum(all_confs) / len(all_confs) if all_confs else 0
                print(f"  [{pct:5.1f}%]  t={ts:6.1f}s  "
                      f"detections: {len(confs)}   running avg conf: {avg:.3f}")

        frame_idx += 1

    cap.release()
    writer.release()

    # Write CSV
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["frame","time_s","count",
                                           "avg_conf","max_conf","all_confs"])
        w.writeheader()
        w.writerows(csv_rows)

    # Write summary
    n_frames_with = sum(1 for c in frame_counts if c > 0)
    avg_conf = sum(all_confs) / len(all_confs) if all_confs else 0
    max_conf = max(all_confs) if all_confs else 0
    avg_det  = sum(frame_counts) / len(frame_counts) if frame_counts else 0

    summary = [
        f"Model            : {model_label}",
        f"Source           : {source.name}",
        f"Frames processed : {len(frame_counts)}",
        f"Frames w/ person : {n_frames_with} / {len(frame_counts)} "
            f"({100*n_frames_with/max(len(frame_counts),1):.1f}%)",
        f"Avg detections   : {avg_det:.2f} / frame",
        f"Total detections : {sum(frame_counts)}",
        f"Avg confidence   : {avg_conf:.4f}",
        f"Max confidence   : {max_conf:.4f}",
    ]
    summary_path = out_dir / "summary.txt"
    summary_path.write_text("\n".join(summary))

    print()
    print("=" * 50)
    for line in summary:
        print(f"  {line}")
    print("=" * 50)
    print(f"\n  Video  → {out_path}")
    print(f"  CSV    → {csv_path}")
    print(f"  Report → {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect persons and annotate output")
    parser.add_argument("--source", required=True, help="Image or video path")
    parser.add_argument("--model",  default="baseline",
                        choices=["baseline", "finetuned"],
                        help="baseline = stock YOLOE-26s | finetuned = trained checkpoint")
    parser.add_argument("--every",  type=int, default=1,
                        help="Process every Nth frame (video only, default=1)")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        sys.exit(f"File not found: {source}")

    model       = load_model(args.model)
    color       = COLORS[args.model]
    model_label = ("Baseline YOLOE-26s (zero-shot)"
                   if args.model == "baseline" else "Fine-tuned YOLOE-26s")

    out_dir = Path("runs/detections") / source.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output folder: {out_dir}\n")

    ext = source.suffix.lower()
    if ext in VIDEO_EXTS:
        process_video(source, model, color, model_label, out_dir, args.every)
    elif ext in IMAGE_EXTS:
        process_image(source, model, color, model_label, out_dir)
    else:
        sys.exit(f"Unsupported format: {ext}")


if __name__ == "__main__":
    main()
