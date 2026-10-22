"""
Step 4 — Baseline evaluation on the TEST split
Compares three models at IoU=0.5 using the corrected test labels:
  1. Zero-shot YOLOE-26s  (set_classes)
  2. COCO YOLO11n/26s     (class 0 = person)
  3. Current production model (optional)

Falls back to predict+manual mAP if model.val() fails (e.g. seg vs detect mismatch).

Usage:
    python scripts/eval_baselines.py [--config configs/train_person.yaml]
                                     [--paths configs/paths.yaml]
"""

import argparse
import sys
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(p, base):
    p = Path(p)
    return p if p.is_absolute() else base / p


# ---------------------------------------------------------------------------
# Manual mAP fallback (predict → compare against labels)
# ---------------------------------------------------------------------------

def compute_metrics_from_predictions(model, img_dir: Path, lbl_dir: Path,
                                     classes: list, conf: float, iou_thr: float):
    """
    Run model.predict on all images in img_dir, compare to ground-truth labels.
    Returns dict with keys: precision, recall, f1, mAP50 (approximate), n_images.

    This is a simplified implementation for the fallback path.
    For production-quality metrics use the val() path.
    """
    import numpy as np
    import cv2

    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    img_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in img_exts)

    tp_total = fp_total = fn_total = 0
    all_precisions = []
    all_recalls    = []

    for img_path in img_paths:
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]

        # Ground truth boxes (xyxy, normalised → absolute)
        gt_boxes = []
        if lbl_path.exists():
            for line in lbl_path.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) == 5 and int(parts[0]) in range(len(classes)):
                    cx, cy, bw, bh = map(float, parts[1:])
                    x1 = (cx - bw / 2) * w
                    y1 = (cy - bh / 2) * h
                    x2 = (cx + bw / 2) * w
                    y2 = (cy + bh / 2) * h
                    gt_boxes.append([x1, y1, x2, y2])

        results = model.predict(source=str(img_path), conf=conf, verbose=False)
        pred_boxes = []
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            for box in r.boxes.xyxy.cpu().numpy():
                pred_boxes.append(box.tolist())

        # Match predictions to GT using IoU threshold
        matched_gt = set()
        tp = fp = 0
        for pb in pred_boxes:
            best_iou = 0.0
            best_idx = -1
            for gi, gb in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pb, gb)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = gi
            if best_iou >= iou_thr and best_idx >= 0:
                tp += 1
                matched_gt.add(best_idx)
            else:
                fp += 1
        fn = len(gt_boxes) - len(matched_gt)

        tp_total += tp
        fp_total += fp
        fn_total += fn

    prec = tp_total / (tp_total + fp_total + 1e-9)
    rec  = tp_total / (tp_total + fn_total + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    return {
        "precision": prec,
        "recall":    rec,
        "f1":        f1,
        "mAP50":     (prec + rec) / 2,   # rough approximation
        "mAP50_95":  None,
        "n_images":  len(img_paths),
        "method":    "predict+manual",
    }


def compute_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


# ---------------------------------------------------------------------------
# Evaluate one model via val() with fallback
# ---------------------------------------------------------------------------

def evaluate_model(model, name: str, dataset_yaml: Path, imgsz: int,
                   img_dir: Path, lbl_dir: Path, classes: list, conf: float):
    print(f"\n  Evaluating: {name}")
    metrics = None
    method = "val()"

    try:
        results = model.val(
            data=str(dataset_yaml),
            split="test",
            imgsz=imgsz,
            conf=conf,
            iou=0.5,
            verbose=False,
        )
        # ultralytics returns a Results object; extract person class metrics
        mp = results.box.mp if hasattr(results.box, "mp") else results.results_dict.get("metrics/precision(B)", None)
        mr = results.box.mr if hasattr(results.box, "mr") else results.results_dict.get("metrics/recall(B)", None)
        map50   = results.box.map50 if hasattr(results.box, "map50") else results.results_dict.get("metrics/mAP50(B)", None)
        map5095 = results.box.map   if hasattr(results.box, "map")   else results.results_dict.get("metrics/mAP50-95(B)", None)
        f1 = (2 * mp * mr / (mp + mr + 1e-9)) if (mp is not None and mr is not None) else None
        metrics = {
            "precision": mp,
            "recall":    mr,
            "f1":        f1,
            "mAP50":     map50,
            "mAP50_95":  map5095,
            "method":    method,
        }
    except Exception as e:
        print(f"    model.val() failed ({e}); falling back to predict+manual metrics")
        metrics = compute_metrics_from_predictions(
            model, img_dir, lbl_dir, classes, conf=conf, iou_thr=0.5
        )

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 4: baseline evaluation")
    parser.add_argument("--config", default="configs/train_person.yaml")
    parser.add_argument("--paths",  default="configs/paths.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cfg  = load_yaml(project_root / args.config)
    pcfg = load_yaml(project_root / args.paths)

    classes    = cfg["classes"]
    imgsz      = cfg["imgsz"]
    conf       = cfg["conf_threshold_default"]
    dataset_dir = resolve_path(pcfg["dataset_dir"], project_root)
    dataset_yaml = dataset_dir / f"{dataset_dir.name}.yaml"
    prod_weights = pcfg.get("production_weights")

    if prod_weights:
        prod_weights = resolve_path(prod_weights, project_root)

    test_img_dir = dataset_dir / "images" / "test"
    test_lbl_dir = dataset_dir / "labels" / "test"

    if not dataset_yaml.exists():
        sys.exit(f"Dataset YAML not found: {dataset_yaml}\nRun Step 3 first.")

    from ultralytics import YOLOE, YOLO

    results_table = {}

    # ------------------------------------------------------------------
    # 1. Zero-shot YOLOE-26s
    # ------------------------------------------------------------------
    print("\n[1/3] Zero-shot YOLOE-26s")
    model_zs = YOLOE("yoloe-26s-seg.pt")
    model_zs.set_classes(classes)
    results_table["YOLOE-26s (zero-shot)"] = evaluate_model(
        model_zs, "YOLOE-26s zero-shot", dataset_yaml, imgsz,
        test_img_dir, test_lbl_dir, classes, conf
    )

    # ------------------------------------------------------------------
    # 2. COCO YOLO26s (class 0 = person)
    # ------------------------------------------------------------------
    print("\n[2/3] COCO-pretrained YOLO11n (person class)")
    try:
        model_coco = YOLO("yolo11n.pt")   # smallest available; swap for yolo26s.pt if you have it
    except Exception:
        model_coco = YOLO("yolov8n.pt")   # ultimate fallback
    results_table["COCO YOLO (pretrained)"] = evaluate_model(
        model_coco, "COCO YOLO pretrained", dataset_yaml, imgsz,
        test_img_dir, test_lbl_dir, classes, conf
    )

    # ------------------------------------------------------------------
    # 3. Production model (optional)
    # ------------------------------------------------------------------
    if prod_weights and Path(prod_weights).exists():
        print(f"\n[3/3] Production model: {prod_weights}")
        model_prod = YOLO(str(prod_weights))
        results_table["Production model"] = evaluate_model(
            model_prod, "Production model", dataset_yaml, imgsz,
            test_img_dir, test_lbl_dir, classes, conf
        )
    else:
        print("\n[3/3] Production model: skipped (not configured or file not found)")

    # ------------------------------------------------------------------
    # Print results table
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("BASELINE RESULTS (TEST split, IoU=0.5)")
    print("=" * 80)
    fmt = "{:<30} {:>10} {:>10} {:>8} {:>10} {:>12} {:>14}"
    print(fmt.format("Model", "Precision", "Recall", "F1", "mAP50", "mAP50-95", "Method"))
    print("-" * 80)
    for name, m in results_table.items():
        def fmt_val(v):
            return f"{v:.4f}" if v is not None else "  n/a  "
        print(fmt.format(
            name[:30],
            fmt_val(m.get("precision")),
            fmt_val(m.get("recall")),
            fmt_val(m.get("f1")),
            fmt_val(m.get("mAP50")),
            fmt_val(m.get("mAP50_95")),
            m.get("method", ""),
        ))

    # Save to YAML for use by Step 6
    baseline_results_path = project_root / "reports" / "baselines.yaml"
    baseline_results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(baseline_results_path, "w") as f:
        yaml.dump(results_table, f)
    print(f"\nBaseline results saved to: {baseline_results_path}")


if __name__ == "__main__":
    main()
