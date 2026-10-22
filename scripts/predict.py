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


def draw_boxes(frame_bgr: np.ndarray, result, label: str, color) -> np.ndarray:
    img = frame_bgr.copy()
    h, w = img.shape[:2]
    count = 0
    if result.boxes is not None and len(result.boxes) > 0:
        for box, c in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"person {c:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            count += 1
    banner = np.zeros((36, w, 3), dtype=np.uint8)
    banner[:] = (40, 40, 40)
    cv2.putText(banner, f"{label}  |  {count} person(s)", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return np.vstack([banner, img]), count


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
    img_base, n_base = draw_boxes(img, r_base, "Baseline (zero-shot)", (0, 200, 80))
    cv2.imwrite(str(out_dir / "baseline.jpg"), img_base)
    print(f"  → {n_base} person(s)")

    if model_ft:
        print("Running fine-tuned ...")
        r_ft = predict_frame(model_ft, img, conf)
        img_ft, n_ft = draw_boxes(img, r_ft, "Fine-tuned YOLOE-26s", (0, 120, 255))
        cv2.imwrite(str(out_dir / "finetuned.jpg"), img_ft)
        print(f"  → {n_ft} person(s)")
        comp = side_by_side(img_base, img_ft)
        cv2.imwrite(str(out_dir / "comparison.jpg"), comp)
        print(f"\nSaved: {out_dir / 'comparison.jpg'}  ← open this")
    else:
        cv2.imwrite(str(out_dir / "comparison.jpg"), img_base)
        print(f"Saved: {out_dir / 'baseline.jpg'}")


# ---------------------------------------------------------------------------
# Video path
# ---------------------------------------------------------------------------

def run_video(source: Path, model_base, model_ft, conf: float,
              out_dir: Path, every: int):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        sys.exit(f"Could not open video: {source}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Output video: side-by-side (2*w + 4 divider, h + 36 banner)
    out_w = (w_orig * 2 + 4) if model_ft else w_orig
    out_h = h_orig + 36
    out_path = out_dir / "comparison.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps / every, (out_w, out_h))

    print(f"Video: {w_orig}x{h_orig} @ {fps:.1f}fps  |  {total} frames  "
          f"|  processing every {every} frame(s)")
    print(f"Output: {out_path}")

    frame_idx = 0
    written   = 0
    base_total = ft_total = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % every == 0:
            r_base = predict_frame(model_base, frame, conf)
            img_base, n_base = draw_boxes(frame, r_base, "Baseline (zero-shot)", (0, 200, 80))
            base_total += n_base

            if model_ft:
                r_ft = predict_frame(model_ft, frame, conf)
                img_ft, n_ft = draw_boxes(frame, r_ft, "Fine-tuned YOLOE-26s", (0, 120, 255))
                ft_total += n_ft
                comp = side_by_side(img_base, img_ft)
            else:
                comp = img_base

            writer.write(comp)
            written += 1

            if written % 30 == 0 or frame_idx == 0:
                pct = 100 * frame_idx / max(total, 1)
                print(f"  [{pct:5.1f}%] frame {frame_idx}/{total}  "
                      f"baseline detections so far: {base_total}"
                      + (f"  finetuned: {ft_total}" if model_ft else ""))

        frame_idx += 1

    cap.release()
    writer.release()

    print(f"\nDone. {written} frames written → {out_path}")
    if model_ft:
        print(f"  Baseline total detections : {base_total}")
        print(f"  Fine-tuned total detections: {ft_total}")


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
