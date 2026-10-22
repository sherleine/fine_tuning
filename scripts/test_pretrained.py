"""
Test stock YOLOE-26s (zero-shot) on COCO person detection.
Prints precision, recall, F1, mAP50, mAP50-95 and saves results to
runs/baseline_results.yaml so finetune_compare.py can load them later.

Usage:
    # Quick smoke-test (128 images, auto-downloads ~7 MB):
    python scripts/test_pretrained.py --data coco128.yaml

    # Real evaluation (5000 images, auto-downloads ~1 GB val set):
    python scripts/test_pretrained.py --data coco.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml


def check_env():
    try:
        import ultralytics
        print(f"ultralytics : {ultralytics.__version__}")
    except ImportError:
        sys.exit("Run: pip install -U ultralytics")

    from packaging.version import Version
    if Version(ultralytics.__version__) < Version("8.4.0"):
        sys.exit("Need ultralytics >= 8.4.0  — run: pip install -U ultralytics")

    try:
        from ultralytics import YOLOE  # noqa: F401
        print("YOLOE       : found  ✓")
    except ImportError:
        sys.exit("YOLOE not in this ultralytics build. pip install -U ultralytics")

    import torch
    print(f"torch       : {torch.__version__}")
    print(f"CUDA        : {torch.version.cuda or 'N/A'}")
    print(f"GPU         : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
    print()


def extract_metrics(results) -> dict:
    """Pull the numbers we care about from a val() result object."""
    box = results.box
    prec   = float(box.mp)    if hasattr(box, "mp")    else None
    rec    = float(box.mr)    if hasattr(box, "mr")    else None
    map50  = float(box.map50) if hasattr(box, "map50") else None
    map    = float(box.map)   if hasattr(box, "map")   else None
    f1 = (2 * prec * rec / (prec + rec + 1e-9)) if (prec and rec) else None
    return {"precision": prec, "recall": rec, "f1": f1, "mAP50": map50, "mAP50_95": map}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="coco128-seg.yaml",
                        help="Dataset YAML. Must have segmentation labels when using the -seg model.\n"
                             "  Quick test : coco128-seg.yaml  (auto-downloads ~7 MB)\n"
                             "  Full eval  : coco-seg.yaml     (auto-downloads ~1 GB)\n"
                             "  Boxes-only : use yoloe-26s.pt (detection model) + coco128.yaml")
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--conf",   type=float, default=0.25)
    args = parser.parse_args()

    check_env()

    from ultralytics import YOLOE

    print("=" * 55)
    print("Loading YOLOE-26s (stock pretrained weights) ...")
    print("=" * 55)

    model = YOLOE("yoloe-26s-seg.pt")
    model.set_classes(["person"])

    print(f"\nEvaluating on: {args.data}  (split=val, imgsz={args.imgsz})")
    print("This will auto-download the dataset if not already cached.")
    if "seg" not in args.data:
        print("[WARN] You're using a detection-only dataset with the seg model.")
        print("       Switch to coco128-seg.yaml or coco-seg.yaml to avoid errors.\n")
    else:
        print()

    results = model.val(
        data=args.data,
        split="val",
        imgsz=args.imgsz,
        conf=args.conf,
        iou=0.5,
        verbose=True,
    )

    m = extract_metrics(results)

    print("\n" + "=" * 55)
    print("BASELINE RESULTS (stock YOLOE-26s, zero-shot)")
    print("=" * 55)
    print(f"  Precision  : {m['precision']:.4f}")
    print(f"  Recall     : {m['recall']:.4f}")
    print(f"  F1         : {m['f1']:.4f}")
    print(f"  mAP50      : {m['mAP50']:.4f}")
    print(f"  mAP50-95   : {m['mAP50_95']:.4f}")
    print("=" * 55)

    # Save for the comparison script
    out = Path("runs/baseline_results.yaml")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        yaml.dump({"baseline_yoloe_zeroshot": m, "data": args.data, "imgsz": args.imgsz}, f)
    print(f"\nSaved to: {out}")
    print("\nNext: run scripts/finetune_compare.py --data", args.data)


if __name__ == "__main__":
    main()
