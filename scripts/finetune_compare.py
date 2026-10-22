"""
Fine-tune YOLOE-26s on COCO (person class) then compare against the baseline.
Loads baseline numbers from runs/baseline_results.yaml (written by test_pretrained.py).

Usage:
    # Quick smoke-test (same 128 images for train+val — just to verify the pipeline):
    python scripts/finetune_compare.py --data coco128.yaml --epochs 5

    # Real run (COCO train → COCO val — auto-downloads ~20 GB):
    python scripts/finetune_compare.py --data coco.yaml --epochs 80

    # Resume / only run eval on an existing checkpoint:
    python scripts/finetune_compare.py --data coco.yaml --weights runs/yoloe_finetune/weights/best.pt --skip-train
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def extract_metrics(results) -> dict:
    box = results.box
    prec  = float(box.mp)    if hasattr(box, "mp")    else None
    rec   = float(box.mr)    if hasattr(box, "mr")    else None
    map50 = float(box.map50) if hasattr(box, "map50") else None
    map   = float(box.map)   if hasattr(box, "map")   else None
    f1 = (2 * prec * rec / (prec + rec + 1e-9)) if (prec and rec) else None
    return {"precision": prec, "recall": rec, "f1": f1, "mAP50": map50, "mAP50_95": map}


def print_comparison(baseline: dict, finetuned: dict):
    def fmt(v):
        return f"{v:.4f}" if v is not None else "  n/a "

    def delta(a, b):
        if a is None or b is None:
            return "     "
        d = b - a
        sign = "+" if d >= 0 else ""
        return f"({sign}{d:.4f})"

    print("\n" + "=" * 65)
    print("COMPARISON:  stock YOLOE-26s  vs  fine-tuned YOLOE-26s")
    print("=" * 65)
    print(f"{'Metric':<14} {'Baseline':>10} {'Fine-tuned':>12} {'Delta':>12}")
    print("-" * 65)

    metrics = [
        ("Precision",  "precision"),
        ("Recall",     "recall"),
        ("F1",         "f1"),
        ("mAP50",      "mAP50"),
        ("mAP50-95",   "mAP50_95"),
    ]
    for label, key in metrics:
        b = baseline.get(key)
        ft = finetuned.get(key)
        print(f"{label:<14} {fmt(b):>10} {fmt(ft):>12} {delta(b, ft):>12}")

    print("=" * 65)

    # Plain-English verdict
    rec_b  = baseline.get("recall")  or 0
    rec_ft = finetuned.get("recall") or 0
    map_b  = baseline.get("mAP50")   or 0
    map_ft = finetuned.get("mAP50")  or 0

    print()
    if map_ft > map_b:
        print(f"✓ Fine-tuning improved mAP50 by {map_ft - map_b:+.4f}")
    elif map_ft < map_b:
        print(f"✗ Fine-tuning reduced mAP50 by {map_ft - map_b:.4f}  "
              "(may need more epochs or data)")
    else:
        print("→ mAP50 unchanged")

    if rec_ft > rec_b:
        print(f"✓ Recall improved by {rec_ft - rec_b:+.4f}  "
              "(model misses fewer people)")
    elif rec_ft < rec_b:
        print(f"✗ Recall dropped by {rec_ft - rec_b:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       default="coco128-seg.yaml",
                        help="Dataset YAML. Seg model (yoloe-26s-seg.pt) needs segmentation labels.\n"
                             "  Quick : coco128-seg.yaml | Full: coco-seg.yaml\n"
                             "  Boxes-only proctoring data: use --model yoloe-26s.pt + plain coco128.yaml")
    parser.add_argument("--model",      default="yoloe-26s-seg.pt",
                        help="Base weights: yoloe-26s-seg.pt (seg data) or yoloe-26s.pt (boxes-only data)")
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--imgsz",      type=int,   default=640)
    parser.add_argument("--batch",      type=int,   default=-1,
                        help="Batch size (-1 = auto)")
    parser.add_argument("--conf",       type=float, default=0.25)
    parser.add_argument("--weights",    default=None,
                        help="Path to existing checkpoint to evaluate (skips training)")
    parser.add_argument("--skip-train", action="store_true",
                        help="Skip training, only run eval on --weights")
    parser.add_argument("--run-name",   default="yoloe_finetune")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Check YOLOEPETrainer
    # ------------------------------------------------------------------
    try:
        from ultralytics.models.yolo.yoloe import YOLOEPETrainer
    except ImportError:
        sys.exit(
            "YOLOEPETrainer not found.\n"
            "Run: pip install -U ultralytics  (need >= 8.4.0)"
        )

    from ultralytics import YOLOE

    best_weights = Path(args.weights) if args.weights else None

    # ------------------------------------------------------------------
    # Fine-tune
    # ------------------------------------------------------------------
    if not args.skip_train:
        print("=" * 55)
        print(f"Fine-tuning YOLOE-26s on {args.data}")
        print(f"  epochs={args.epochs}  imgsz={args.imgsz}  batch={args.batch}")
        print("=" * 55)

        model = YOLOE(args.model)

        try:
            model.train(
                data      = args.data,
                epochs    = args.epochs,
                imgsz     = args.imgsz,
                batch     = args.batch,
                patience  = max(10, args.epochs // 4),
                project   = "runs",
                name      = args.run_name,
                exist_ok  = True,
                seed      = 42,
                trainer   = YOLOEPETrainer,
                classes   = [0],   # person only
            )
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
                print("\n[OOM] Retrying with batch=8 ...")
                torch.cuda.empty_cache()
                model.train(
                    data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                    batch=8, patience=max(10, args.epochs // 4),
                    project="runs", name=args.run_name, exist_ok=True,
                    seed=42, trainer=YOLOEPETrainer, classes=[0],
                )
            else:
                raise

        # Find best weights — search recursively because ultralytics may nest under runs/<task>/
        candidates = sorted(Path("runs").glob(f"**/{args.run_name}*/weights/best.pt"))
        if not candidates:
            candidates = sorted(Path("runs").glob(f"**/{args.run_name}*/weights/last.pt"))
        if not candidates:
            # Last resort: any best.pt anywhere under runs/
            candidates = sorted(Path("runs").glob("**/weights/best.pt"))
        if not candidates:
            sys.exit("Training finished but best.pt not found under runs/. Check for errors above.")
        best_weights = candidates[-1]
        print(f"\nBest weights: {best_weights}")

    else:
        if not best_weights or not best_weights.exists():
            sys.exit("--skip-train requires --weights <path/to/best.pt>")
        print(f"Skipping training. Evaluating: {best_weights}")

    # ------------------------------------------------------------------
    # Evaluate fine-tuned model
    # ------------------------------------------------------------------
    print(f"\nEvaluating fine-tuned model on {args.data} (split=val) ...")
    model_ft = YOLOE(str(best_weights))
    if hasattr(model_ft, "set_classes"):
        model_ft.set_classes(["person"])

    results_ft = model_ft.val(
        data  = args.data,
        split = "val",
        imgsz = args.imgsz,
        conf  = args.conf,
        iou   = 0.5,
        verbose = False,
    )
    ft_metrics = extract_metrics(results_ft)

    # ------------------------------------------------------------------
    # Load baseline (run test_pretrained.py first)
    # ------------------------------------------------------------------
    baseline_path = Path("runs/baseline_results.yaml")
    if baseline_path.exists():
        saved = load_yaml(baseline_path)
        baseline_metrics = saved.get("baseline_yoloe_zeroshot", {})
    else:
        print("\n[WARN] runs/baseline_results.yaml not found.")
        print("  Run scripts/test_pretrained.py first to get baseline numbers.")
        print("  Showing fine-tuned results only.\n")
        baseline_metrics = {}

    # ------------------------------------------------------------------
    # Compare
    # ------------------------------------------------------------------
    if baseline_metrics:
        print_comparison(baseline_metrics, ft_metrics)
    else:
        print("\nFine-tuned results:")
        for k, v in ft_metrics.items():
            print(f"  {k:<12}: {v:.4f}" if v is not None else f"  {k:<12}: n/a")

    # Save combined results
    out = Path("runs/comparison_results.yaml")
    with open(out, "w") as f:
        yaml.dump({
            "data": args.data,
            "imgsz": args.imgsz,
            "finetuned_weights": str(best_weights),
            "baseline": baseline_metrics,
            "finetuned": ft_metrics,
        }, f)
    print(f"\nFull results saved to: {out}")


if __name__ == "__main__":
    main()
