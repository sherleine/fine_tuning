"""
Run both models on an image or video and save side-by-side annotated results.

Usage:
    python scripts/predict.py --source path/to/photo.jpg
    python scripts/predict.py --source path/to/video.mp4
    python scripts/predict.py --source path/to/video.mp4 --every 5   # every 5th frame

Outputs (image):
    runs/predict/baseline.jpg      — stock YOLOE-26s result
    runs/predict/finetuned.jpg     — fine-tuned result
    runs/predict/comparison.jpg    — side by side

Outputs (video):
    runs/predict/comparison.mp4    — side-by-side comparison video
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".wmv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def find_best_weights(run_name="yoloe_finetune") -> Path | None:
    candidates = sorted(Path("runs").glob(f"**/{run_name}*/weights/best.pt"))
    if not candidates:
        candidates = sorted(Path("runs").glob("**/weights/best.pt"))
    return candidates[-1] if candidates else None


def predict_frame(model, frame_bgr: np.ndarray, conf: float):
    results = model.predict(source=frame_bgr, conf=conf, verbose=False)
    return results[0]


def draw_boxes(frame_bgr: np.ndarray, result, label: str, color) -> tuple[np.ndarray, int, list]:
    """Returns annotated frame, detection count, and list of confidence scores."""
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    confs = []

    if result.boxes is not None and len(result.boxes) > 0:
        for box, c in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            confs.append(float(c))
            # Box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            # Confidence badge (filled background for readability)
            badge = f"{c:.2f}"
            (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            by = max(y1 - 4, th + 4)
            cv2.rectangle(img, (x1, by - th - 4), (x1 + tw + 6, by + 2), color, -1)
            cv2.putText(img, badge, (x1 + 3, by - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    count = len(confs)
    avg_conf = sum(confs) / count if confs else 0.0
    max_conf = max(confs) if confs else 0.0

    # Stats banner (2 lines)
    banner = np.zeros((52, w, 3), dtype=np.uint8)
    banner[:] = (30, 30, 30)
    cv2.putText(banner, label, (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    stats = (f"detected: {count}   avg conf: {avg_conf:.3f}   max conf: {max_conf:.3f}"
             if count else "detected: 0")
    cv2.putText(banner, stats, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 120), 1)

    return np.vstack([banner, img]), count, confs


def side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    h1, w1 = left.shape[:2]
    h2, w2 = right.shape[:2]
    if h1 != h2:
        right = cv2.resize(right, (int(w2 * h1 / h2), h1))
    div = np.full((h1, 4, 3), 200, dtype=np.uint8)
    return np.hstack([left, div, right])


# ---------------------------------------------------------------------------
# Image path
# ---------------------------------------------------------------------------

def run_image(source: Path, model_base, model_ft, conf: float, out_dir: Path):
    img = cv2.imread(str(source))
    if img is None:
        sys.exit(f"Could not read image: {source}")

    print("Running baseline ...")
    r_base = predict_frame(model_base, img, conf)
    img_base, n_base, confs_base = draw_boxes(img, r_base, "Baseline (zero-shot)", (0, 200, 80))
    cv2.imwrite(str(out_dir / "baseline.jpg"), img_base)

    if model_ft:
        print("Running fine-tuned ...")
        r_ft = predict_frame(model_ft, img, conf)
        img_ft, n_ft, confs_ft = draw_boxes(img, r_ft, "Fine-tuned YOLOE-26s", (0, 120, 255))
        cv2.imwrite(str(out_dir / "finetuned.jpg"), img_ft)
        comp = side_by_side(img_base, img_ft)
        cv2.imwrite(str(out_dir / "comparison.jpg"), comp)

        print()
        print("=" * 52)
        print(f"{'':20} {'Baseline':>14} {'Fine-tuned':>14}")
        print("-" * 52)
        print(f"{'Detections':<20} {n_base:>14} {n_ft:>14}")
        avg_b = sum(confs_base)/len(confs_base) if confs_base else 0
        avg_f = sum(confs_ft)/len(confs_ft) if confs_ft else 0
        max_b = max(confs_base) if confs_base else 0
        max_f = max(confs_ft) if confs_ft else 0
        print(f"{'Avg confidence':<20} {avg_b:>14.4f} {avg_f:>14.4f}")
        print(f"{'Max confidence':<20} {max_b:>14.4f} {max_f:>14.4f}")
        if confs_base:
            print(f"{'Per-box confs':<20} {'  '.join(f'{c:.3f}' for c in confs_base):>14}")
        if confs_ft:
            print(f"{'':20} {'  '.join(f'{c:.3f}' for c in confs_ft):>14}")
        print("=" * 52)
        print(f"\nSaved: {out_dir / 'comparison.jpg'}  ← open this")
    else:
        cv2.imwrite(str(out_dir / "comparison.jpg"), img_base)
        print(f"Detections: {n_base}  confs: {confs_base}")
        print(f"Saved: {out_dir / 'baseline.jpg'}")


# ---------------------------------------------------------------------------
# Video path
# ---------------------------------------------------------------------------

def run_video(source: Path, model_base, model_ft, conf: float,
              out_dir: Path, every: int):
    import csv

    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {source}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_w  = (w_orig * 2 + 4) if model_ft else w_orig
    out_h  = h_orig + 52          # 52px banner
    out_path  = out_dir / "comparison.mp4"
    csv_path  = out_dir / "detections.csv"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps / every, (out_w, out_h))

    print(f"Video     : {w_orig}x{h_orig} @ {fps:.1f} fps  |  {total} total frames")
    print(f"Processing: every {every} frame(s)  →  ~{total//every} output frames")
    print(f"Output    : {out_path}")
    print(f"CSV log   : {csv_path}\n")

    frame_idx  = 0
    written    = 0
    rows       = []   # for CSV

    # Running stats
    base_counts, base_all_confs = [], []
    ft_counts,   ft_all_confs   = [], []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every == 0:
            ts = frame_idx / fps   # timestamp in seconds

            r_base = predict_frame(model_base, frame, conf)
            img_base, n_base, confs_base = draw_boxes(
                frame, r_base, "Baseline (zero-shot)", (0, 200, 80))
            base_counts.append(n_base)
            base_all_confs.extend(confs_base)

            row = {
                "frame": frame_idx,
                "time_s": f"{ts:.3f}",
                "baseline_count": n_base,
                "baseline_confs": ";".join(f"{c:.4f}" for c in confs_base),
                "baseline_avg_conf": f"{sum(confs_base)/len(confs_base):.4f}" if confs_base else "0",
            }

            if model_ft:
                r_ft = predict_frame(model_ft, frame, conf)
                img_ft, n_ft, confs_ft = draw_boxes(
                    frame, r_ft, "Fine-tuned YOLOE-26s", (0, 120, 255))
                ft_counts.append(n_ft)
                ft_all_confs.extend(confs_ft)
                comp = side_by_side(img_base, img_ft)
                row.update({
                    "finetuned_count": n_ft,
                    "finetuned_confs": ";".join(f"{c:.4f}" for c in confs_ft),
                    "finetuned_avg_conf": f"{sum(confs_ft)/len(confs_ft):.4f}" if confs_ft else "0",
                })
            else:
                comp = img_base

            rows.append(row)
            writer.write(comp)
            written += 1

            if written % 25 == 0 or frame_idx == 0:
                pct = 100 * frame_idx / max(total, 1)
                base_avg = sum(base_all_confs)/len(base_all_confs) if base_all_confs else 0
                line = (f"  [{pct:5.1f}%] t={ts:6.1f}s  "
                        f"baseline: {n_base} det (avg conf {base_avg:.3f})")
                if model_ft:
                    ft_avg = sum(ft_all_confs)/len(ft_all_confs) if ft_all_confs else 0
                    line += f"  |  finetuned: {n_ft} det (avg conf {ft_avg:.3f})"
                print(line)

        frame_idx += 1

    cap.release()
    writer.release()

    # Write CSV
    if rows:
        fields = list(rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)

    # Final summary
    print()
    print("=" * 62)
    print(f"{'SUMMARY':^62}")
    print("=" * 62)

    def stats_line(label, counts, confs):
        if not counts:
            return
        avg_c = sum(counts) / len(counts)
        avg_k = sum(confs)  / len(confs)  if confs else 0
        max_k = max(confs)  if confs else 0
        frames_with = sum(1 for x in counts if x > 0)
        print(f"  {label}")
        print(f"    Frames with person    : {frames_with} / {len(counts)}  "
              f"({100*frames_with/len(counts):.1f}%)")
        print(f"    Avg detections/frame  : {avg_c:.2f}")
        print(f"    Avg confidence        : {avg_k:.4f}")
        print(f"    Max confidence        : {max_k:.4f}")
        print(f"    Total detections      : {sum(counts)}")

    stats_line("Baseline (zero-shot)", base_counts, base_all_confs)
    if model_ft:
        print()
        stats_line("Fine-tuned YOLOE-26s", ft_counts, ft_all_confs)

    print("=" * 62)
    print(f"\n  Video   → {out_path}")
    print(f"  CSV log → {csv_path}  (frame-by-frame confidence scores)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compare baseline vs fine-tuned on image or video")
    parser.add_argument("--source",   required=True, help="Image or video path")
    parser.add_argument("--conf",     type=float, default=0.25)
    parser.add_argument("--every",    type=int,   default=1,
                        help="Process every Nth frame (video only). Use 3-5 to speed things up.")
    parser.add_argument("--run-name", default="yoloe_finetune")
    args = parser.parse_args()

    source = Path(args.source)
    if not source.exists():
        sys.exit(f"File not found: {source}")

    from ultralytics import YOLOE

    out_dir = Path("runs/predict")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load baseline
    print("Loading baseline (stock YOLOE-26s) ...")
    model_base = YOLOE("yoloe-26s-seg.pt")
    model_base.set_classes(["person"])

    # Load fine-tuned (optional)
    best_weights = find_best_weights(args.run_name)
    model_ft = None
    if best_weights:
        print(f"Loading fine-tuned model: {best_weights}")
        model_ft = YOLOE(str(best_weights))
        # YOLOEPETrainer bakes class embeddings into weights — do NOT call set_classes()
    else:
        print("[WARN] No fine-tuned weights found — showing baseline only.")

    ext = source.suffix.lower()
    if ext in VIDEO_EXTS:
        run_video(source, model_base, model_ft, args.conf, out_dir, args.every)
    elif ext in IMAGE_EXTS:
        run_image(source, model_base, model_ft, args.conf, out_dir)
    else:
        sys.exit(f"Unsupported file type: {ext}\nSupported: {IMAGE_EXTS | VIDEO_EXTS}")


if __name__ == "__main__":
    main()
