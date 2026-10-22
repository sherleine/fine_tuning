"""
Step 6 — Full evaluation and comparison report
Compares all models on the TEST split with per-slice analysis.

Models evaluated:
  1. Zero-shot YOLOE-26s
  2. COCO-pretrained YOLO
  3. Production model (if configured)
  4. Fine-tuned YOLOE full
  5. Fine-tuned YOLOE frozen backbone

Threshold selection: tune on VAL to maximise F1 (or hit precision >= 0.90),
then apply unchanged on TEST.

Output:
  reports/person_pilot.md      — markdown report with tables
  reports/fp_images/           — 20 worst false positives
  reports/fn_images/           — 20 worst false negatives

Usage:
    python scripts/evaluate.py [--config configs/train_person.yaml]
                               [--paths configs/paths.yaml]
"""

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(p, base):
    p = Path(p)
    return p if p.is_absolute() else base / p


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Box:
    x1: float; y1: float; x2: float; y2: float
    conf: float = 1.0

    @property
    def area(self):
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    @property
    def cx(self):
        return (self.x1 + self.x2) / 2

    @property
    def cy(self):
        return (self.y1 + self.y2) / 2


@dataclass
class FrameResult:
    path: Path
    gt_boxes: list[Box]       = field(default_factory=list)
    pred_boxes: list[Box]     = field(default_factory=list)
    camera: str               = "unknown"
    session_id: str           = "unknown"
    mean_luma: float          = 0.0
    tp: int                   = 0
    fp: int                   = 0
    fn: int                   = 0


# ---------------------------------------------------------------------------
# IoU
# ---------------------------------------------------------------------------

def iou(a: Box, b: Box) -> float:
    ix1 = max(a.x1, b.x1); iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2); iy2 = min(a.y2, b.y2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    return inter / (a.area + b.area - inter)


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_gt_boxes(lbl_path: Path, img_w: int, img_h: int) -> list[Box]:
    boxes = []
    if not lbl_path.exists():
        return boxes
    for line in lbl_path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) == 5:
            _, cx, cy, bw, bh = map(float, parts)
            x1 = (cx - bw / 2) * img_w
            y1 = (cy - bh / 2) * img_h
            x2 = (cx + bw / 2) * img_w
            y2 = (cy + bh / 2) * img_h
            boxes.append(Box(x1, y1, x2, y2))
    return boxes


# ---------------------------------------------------------------------------
# Threshold tuning on VAL split
# ---------------------------------------------------------------------------

def tune_threshold(model, val_img_dir: Path, val_lbl_dir: Path,
                   confs=None, iou_thr=0.5, min_precision=0.90) -> float:
    if confs is None:
        confs = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70]

    img_paths = sorted(p for p in val_img_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    best_conf = 0.25
    best_f1   = -1.0

    for conf in confs:
        tp = fp = fn = 0
        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            gt  = load_gt_boxes(val_lbl_dir / (img_path.stem + ".txt"), w, h)
            res = model.predict(source=str(img_path), conf=conf, verbose=False)
            r = res[0]
            preds = []
            if r.boxes is not None:
                for box, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                    preds.append(Box(*box, conf=float(c)))

            matched_gt = set()
            for pb in preds:
                best_iou = 0.0; best_i = -1
                for i, gb in enumerate(gt):
                    if i in matched_gt: continue
                    v = iou(pb, gb)
                    if v > best_iou: best_iou = v; best_i = i
                if best_iou >= iou_thr and best_i >= 0:
                    tp += 1; matched_gt.add(best_i)
                else:
                    fp += 1
            fn += len(gt) - len(matched_gt)

        prec = tp / (tp + fp + 1e-9)
        rec  = tp / (tp + fn + 1e-9)
        f1   = 2 * prec * rec / (prec + rec + 1e-9)

        # Prefer threshold that maximises F1, subject to precision >= min_precision
        if prec >= min_precision and f1 > best_f1:
            best_f1   = f1
            best_conf = conf

    # If no conf met precision constraint, fall back to pure F1 maximisation
    if best_f1 < 0:
        for conf in confs:
            tp = fp = fn = 0
            for img_path in img_paths:
                img = cv2.imread(str(img_path))
                h, w = img.shape[:2]
                gt  = load_gt_boxes(val_lbl_dir / (img_path.stem + ".txt"), w, h)
                res = model.predict(source=str(img_path), conf=conf, verbose=False)
                r = res[0]
                preds = []
                if r.boxes is not None:
                    for box, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                        preds.append(Box(*box, conf=float(c)))
                matched_gt = set()
                for pb in preds:
                    best_iou_v = 0.0; best_i = -1
                    for i, gb in enumerate(gt):
                        if i in matched_gt: continue
                        v = iou(pb, gb)
                        if v > best_iou_v: best_iou_v = v; best_i = i
                    if best_iou_v >= iou_thr and best_i >= 0:
                        tp += 1; matched_gt.add(best_i)
                    else:
                        fp += 1
                fn += len(gt) - len(matched_gt)
            prec = tp / (tp + fp + 1e-9)
            rec  = tp / (tp + fn + 1e-9)
            f1   = 2 * prec * rec / (prec + rec + 1e-9)
            if f1 > best_f1:
                best_f1 = f1; best_conf = conf

    return best_conf


# ---------------------------------------------------------------------------
# Full evaluation on a split
# ---------------------------------------------------------------------------

def evaluate_split(model, img_dir: Path, lbl_dir: Path,
                   conf: float, iou_thr: float,
                   frame_meta: dict,
                   low_light_thr: float) -> list[FrameResult]:
    img_paths = sorted(p for p in img_dir.iterdir()
                       if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    results = []

    for img_path in img_paths:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean_luma = float(gray.mean())

        gt = load_gt_boxes(lbl_dir / (img_path.stem + ".txt"), w, h)

        r_list = model.predict(source=str(img_path), conf=conf, verbose=False)
        r = r_list[0]
        preds = []
        if r.boxes is not None:
            for box, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                preds.append(Box(*box, conf=float(c)))

        matched_gt = set()
        tp = fp = 0
        for pb in preds:
            best_iou_v = 0.0; best_i = -1
            for i, gb in enumerate(gt):
                if i in matched_gt: continue
                v = iou(pb, gb)
                if v > best_iou_v: best_iou_v = v; best_i = i
            if best_iou_v >= iou_thr and best_i >= 0:
                tp += 1; matched_gt.add(best_i)
            else:
                fp += 1
        fn = len(gt) - len(matched_gt)

        meta = frame_meta.get(img_path.stem, {})
        fr = FrameResult(
            path       = img_path,
            gt_boxes   = gt,
            pred_boxes = preds,
            camera     = meta.get("camera", "unknown"),
            session_id = meta.get("session_id", "unknown"),
            mean_luma  = mean_luma,
            tp=tp, fp=fp, fn=fn,
        )
        results.append(fr)

    return results


# ---------------------------------------------------------------------------
# Metric aggregation
# ---------------------------------------------------------------------------

def aggregate(results: list[FrameResult]) -> dict:
    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)

    # Count accuracy (predicted count == true count)
    count_correct = sum(1 for r in results if len(r.pred_boxes) == len(r.gt_boxes))
    count_acc = count_correct / max(len(results), 1)

    # FP/1000 on empty frames
    empty_results = [r for r in results if len(r.gt_boxes) == 0]
    fp_on_empty = sum(r.fp for r in empty_results)
    fp_per_1k = (fp_on_empty / max(len(empty_results), 1)) * 1000

    return {
        "precision": prec, "recall": rec, "f1": f1,
        "tp": tp, "fp": fp, "fn": fn,
        "count_accuracy": count_acc,
        "fp_per_1k_empty": fp_per_1k,
        "n_frames": len(results),
        "n_empty": len(empty_results),
    }


def slice_metrics(results: list[FrameResult], label: str,
                  low_light_thr: float) -> dict:
    slices = {}

    # By camera
    for cam in set(r.camera for r in results):
        slices[f"camera={cam}"] = aggregate([r for r in results if r.camera == cam])

    # Multi-person frames
    slices["multi_person (gt>=2)"] = aggregate([r for r in results if len(r.gt_boxes) >= 2])

    # Small / partial people: smallest GT box in bottom third of area range
    # Approximate: box area < 0.05 of image area (relative coords → area < 0.05)
    small_results = []
    for r in results:
        if any((b.area / (r.path.stat().st_size or 1)) < 0.05 for b in r.gt_boxes):
            small_results.append(r)
    # Better approximation without loading the image again: label area fraction
    small_results2 = []
    for r in results:
        lbl = r.path.parent.parent.parent / "labels" / "test" / (r.path.stem + ".txt")
        if lbl.exists():
            for line in lbl.read_text().strip().splitlines():
                parts = line.split()
                if len(parts) == 5:
                    bw, bh = float(parts[3]), float(parts[4])
                    if bw * bh < 0.05:
                        small_results2.append(r)
                        break
    slices["small_partial (box_area<0.05)"] = aggregate(small_results2) if small_results2 else {}

    # Low light
    slices[f"low_light (luma<{low_light_thr})"] = aggregate(
        [r for r in results if r.mean_luma < low_light_thr]
    )

    return slices


# ---------------------------------------------------------------------------
# Annotated image saving
# ---------------------------------------------------------------------------

def save_annotated(results: list[FrameResult], out_dir: Path, n: int,
                   error_type: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sort by worst (most errors)
    if error_type == "fp":
        ranked = sorted(results, key=lambda r: r.fp, reverse=True)
    else:
        ranked = sorted(results, key=lambda r: r.fn, reverse=True)

    saved = 0
    for r in ranked:
        if saved >= n:
            break
        count = r.fp if error_type == "fp" else r.fn
        if count == 0:
            continue
        img = cv2.imread(str(r.path))
        if img is None:
            continue

        h, w = img.shape[:2]
        # Draw GT (green)
        for b in r.gt_boxes:
            cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)),
                          (0, 200, 0), 2)
        # Draw preds (red=FP, blue=TP)
        for b in r.pred_boxes:
            col = (0, 0, 200)  # red = prediction
            cv2.rectangle(img, (int(b.x1), int(b.y1)), (int(b.x2), int(b.y2)),
                          col, 2)
            cv2.putText(img, f"{b.conf:.2f}", (int(b.x1), int(b.y1) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
        label = f"{error_type.upper()}={count}"
        cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 2)
        out_path = out_dir / f"{r.path.stem}_{error_type}{count}.jpg"
        cv2.imwrite(str(out_path), img)
        saved += 1


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------

def write_markdown_report(model_results: dict, slices_by_model: dict,
                          conf_used: dict, report_path: Path):
    lines = [
        "# Person Detection Pilot — Model Comparison Report",
        "",
        "## Overview",
        "",
        "| Model | Conf | Precision | Recall | F1 | mAP50 | Count Acc | FP/1k empty |",
        "|-------|------|-----------|--------|----|-------|-----------|-------------|",
    ]

    for name, m in model_results.items():
        conf = conf_used.get(name, "?")
        lines.append(
            f"| {name} | {conf} "
            f"| {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
            f"| {m.get('mAP50', '—')} "
            f"| {m['count_accuracy']:.3f} | {m['fp_per_1k_empty']:.1f} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Per-Slice Recall",
        "",
    ]

    all_slices = sorted({k for s in slices_by_model.values() for k in s})
    header = "| Slice | " + " | ".join(model_results.keys()) + " |"
    sep    = "|-------|" + "|".join(["-------"] * len(model_results)) + "|"
    lines += [header, sep]

    for sl in all_slices:
        row = f"| {sl} |"
        for name in model_results:
            sm = slices_by_model.get(name, {}).get(sl, {})
            rec = sm.get("recall")
            row += f" {rec:.3f} |" if rec is not None else " — |"
        lines.append(row)

    lines += [
        "",
        "---",
        "",
        "## Conclusion",
        "",
        "> Fill this in after reviewing the numbers above. Key questions:",
        "> 1. Does fine-tuned YOLOE-26s beat both COCO YOLO and the production model on person recall?",
        "> 2. Is recall improvement largest for partial/small people and multi-person frames?",
        "> 3. Does the improvement come without a false-positive increase?",
        "> 4. What FPS does the best model achieve? Is it acceptable for real-time proctoring?",
        "",
        "*(See `reports/fp_images/` and `reports/fn_images/` for annotated failure cases.)*",
    ]

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print(f"Report written to: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 6: full evaluation")
    parser.add_argument("--config", default="configs/train_person.yaml")
    parser.add_argument("--paths",  default="configs/paths.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cfg  = load_yaml(project_root / args.config)
    pcfg = load_yaml(project_root / args.paths)

    classes        = cfg["classes"]
    imgsz          = cfg["imgsz"]
    iou_thr        = cfg["iou_threshold"]
    low_light_thr  = cfg["low_light_luma_threshold"]
    dataset_dir    = resolve_path(pcfg["dataset_dir"], project_root)
    reports_dir    = resolve_path(pcfg["reports_dir"], project_root)

    val_img_dir   = dataset_dir / "images" / "val"
    val_lbl_dir   = dataset_dir / "labels" / "val"
    test_img_dir  = dataset_dir / "images" / "test"
    test_lbl_dir  = dataset_dir / "labels" / "test"

    weights_registry = project_root / "runs" / "finetuned_weights.yaml"
    prod_weights     = pcfg.get("production_weights")

    # Load manifest for frame metadata
    manifest_path = resolve_path(pcfg["frames_dir"], project_root) / "manifest.csv"
    frame_meta: dict[str, dict] = {}
    if manifest_path.exists():
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                frame_meta[Path(row["frame_path"]).stem] = row

    from ultralytics import YOLOE, YOLO

    # ------------------------------------------------------------------
    # Build model registry
    # ------------------------------------------------------------------
    models_to_eval: list[tuple[str, object]] = []

    # 1. Zero-shot YOLOE
    m_zs = YOLOE("yoloe-26s-seg.pt")
    m_zs.set_classes(classes)
    models_to_eval.append(("YOLOE-26s zero-shot", m_zs))

    # 2. COCO YOLO
    try:
        m_coco = YOLO("yolo11n.pt")
    except Exception:
        m_coco = YOLO("yolov8n.pt")
    models_to_eval.append(("COCO YOLO pretrained", m_coco))

    # 3. Production model
    if prod_weights and Path(prod_weights).exists():
        m_prod = YOLO(str(prod_weights))
        models_to_eval.append(("Production model", m_prod))

    # 4 & 5. Fine-tuned models
    if weights_registry.exists():
        fw = load_yaml(weights_registry)
        if fw.get("full") and Path(fw["full"]).exists():
            m_full = YOLOE(fw["full"])
            models_to_eval.append(("YOLOE-26s fine-tuned (full)", m_full))
        if fw.get("frozen") and Path(fw["frozen"]).exists():
            m_frz = YOLOE(fw["frozen"])
            models_to_eval.append(("YOLOE-26s fine-tuned (frozen)", m_frz))
    else:
        print("[WARN] finetuned_weights.yaml not found. Run Step 5 first.")

    # ------------------------------------------------------------------
    # Tune thresholds on VAL, evaluate on TEST
    # ------------------------------------------------------------------
    model_results   = {}
    slices_by_model = {}
    conf_used       = {}

    for name, model in models_to_eval:
        print(f"\n[Threshold tuning on VAL] {name}")
        conf = tune_threshold(model, val_img_dir, val_lbl_dir, iou_thr=iou_thr)
        conf_used[name] = conf
        print(f"  Chosen conf={conf:.2f}")

        print(f"[Evaluating on TEST] {name}")
        test_results = evaluate_split(
            model, test_img_dir, test_lbl_dir,
            conf=conf, iou_thr=iou_thr,
            frame_meta=frame_meta,
            low_light_thr=low_light_thr,
        )

        model_results[name]   = aggregate(test_results)
        slices_by_model[name] = slice_metrics(test_results, name, low_light_thr)

        # Save annotated failure cases for the first fine-tuned model
        if "fine-tuned (full)" in name:
            save_annotated(test_results, reports_dir / "fp_images", 20, "fp")
            save_annotated(test_results, reports_dir / "fn_images", 20, "fn")

    # ------------------------------------------------------------------
    # Print summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("TEST RESULTS")
    print("=" * 90)
    fmt = "{:<36} {:>5} {:>10} {:>8} {:>7} {:>10} {:>12}"
    print(fmt.format("Model", "Conf", "Precision", "Recall", "F1", "CountAcc", "FP/1k_empty"))
    print("-" * 90)
    for name, m in model_results.items():
        print(fmt.format(
            name[:36],
            f"{conf_used.get(name, '?')}",
            f"{m['precision']:.4f}",
            f"{m['recall']:.4f}",
            f"{m['f1']:.4f}",
            f"{m['count_accuracy']:.4f}",
            f"{m['fp_per_1k_empty']:.1f}",
        ))

    # ------------------------------------------------------------------
    # Write markdown report
    # ------------------------------------------------------------------
    report_path = reports_dir / "person_pilot.md"
    write_markdown_report(model_results, slices_by_model, conf_used, report_path)


if __name__ == "__main__":
    main()
