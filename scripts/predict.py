"""
Run both models on a single image and save side-by-side annotated results.

Usage:
    python scripts/predict.py --image path/to/your/photo.jpg

Outputs:
    runs/predict/baseline.jpg      — stock YOLOE-26s result
    runs/predict/finetuned.jpg     — fine-tuned result
    runs/predict/comparison.jpg    — side by side
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


def find_best_weights(run_name="yoloe_finetune") -> Path | None:
    candidates = sorted(Path("runs").glob(f"**/{run_name}*/weights/best.pt"))
    if not candidates:
        candidates = sorted(Path("runs").glob("**/weights/best.pt"))
    return candidates[-1] if candidates else None


def run_predict(model, image_path: Path, conf: float = 0.25):
    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    return results[0]


def draw_boxes(image_bgr: np.ndarray, result, label: str, color=(0, 200, 0)) -> np.ndarray:
    img = image_bgr.copy()
    h, w = img.shape[:2]

    boxes = result.boxes
    count = 0
    if boxes is not None and len(boxes) > 0:
        for box, conf in zip(boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, f"person {conf:.2f}", (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            count += 1

    # Header banner
    banner_h = 36
    banner = np.zeros((banner_h, w, 3), dtype=np.uint8)
    banner[:] = (40, 40, 40)
    text = f"{label}  |  {count} person(s) detected"
    cv2.putText(banner, text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return np.vstack([banner, img])


def main():
    parser = argparse.ArgumentParser(description="Predict with baseline vs fine-tuned model")
    parser.add_argument("--image",    required=True, help="Path to input image")
    parser.add_argument("--conf",     type=float, default=0.25)
    parser.add_argument("--run-name", default="yoloe_finetune")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        sys.exit(f"Image not found: {image_path}")

    from ultralytics import YOLOE

    out_dir = Path("runs/predict")
    out_dir.mkdir(parents=True, exist_ok=True)

    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        sys.exit(f"Could not read image: {image_path}")

    # ------------------------------------------------------------------
    # Baseline: stock YOLOE-26s
    # ------------------------------------------------------------------
    print("Running baseline (stock YOLOE-26s) ...")
    model_base = YOLOE("yoloe-26s-seg.pt")
    model_base.set_classes(["person"])
    result_base = run_predict(model_base, image_path, args.conf)
    n_base = len(result_base.boxes) if result_base.boxes is not None else 0

    img_base = draw_boxes(img_bgr, result_base, "Baseline (zero-shot)", color=(0, 200, 80))
    cv2.imwrite(str(out_dir / "baseline.jpg"), img_base)
    print(f"  → {n_base} person(s) detected")

    # ------------------------------------------------------------------
    # Fine-tuned
    # ------------------------------------------------------------------
    best_weights = find_best_weights(args.run_name)
    if best_weights is None:
        print("\n[WARN] No fine-tuned weights found. Run finetune_compare.py first.")
        print("Showing baseline result only.")
        cv2.imwrite(str(out_dir / "comparison.jpg"), img_base)
        print(f"\nSaved: {out_dir / 'baseline.jpg'}")
        return

    print(f"Running fine-tuned model ({best_weights.parent.parent.name}) ...")
    model_ft = YOLOE(str(best_weights))
    if hasattr(model_ft, "set_classes"):
        model_ft.set_classes(["person"])
    result_ft = run_predict(model_ft, image_path, args.conf)
    n_ft = len(result_ft.boxes) if result_ft.boxes is not None else 0

    img_ft = draw_boxes(img_bgr, result_ft, "Fine-tuned YOLOE-26s", color=(0, 120, 255))
    cv2.imwrite(str(out_dir / "finetuned.jpg"), img_ft)
    print(f"  → {n_ft} person(s) detected")

    # ------------------------------------------------------------------
    # Side-by-side comparison
    # ------------------------------------------------------------------
    # Resize both to same height if they differ (they shouldn't, but just in case)
    h1, w1 = img_base.shape[:2]
    h2, w2 = img_ft.shape[:2]
    if h1 != h2:
        img_ft = cv2.resize(img_ft, (int(w2 * h1 / h2), h1))

    divider = np.full((h1, 4, 3), 200, dtype=np.uint8)  # white separator line
    comparison = np.hstack([img_base, divider, img_ft])
    cv2.imwrite(str(out_dir / "comparison.jpg"), comparison)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 50)
    print(f"Baseline   : {n_base} person(s)")
    print(f"Fine-tuned : {n_ft} person(s)")
    print("=" * 50)
    print(f"\nImages saved to {out_dir}/")
    print(f"  baseline.jpg    — stock model")
    print(f"  finetuned.jpg   — fine-tuned model")
    print(f"  comparison.jpg  — side by side  ← open this one")


if __name__ == "__main__":
    main()
