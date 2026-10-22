"""
Step 2 — Pre-labelling with zero-shot YOLOE-26s
Runs YOLOE on extracted frames and writes YOLO-format label files
at a low confidence threshold so labels can be corrected by hand.

Usage:
    python scripts/prelabel.py [--conf 0.25]
                               [--config configs/train_person.yaml]
                               [--paths configs/paths.yaml]

After this script finishes, import the output folder into CVAT or Label Studio
using the YOLO 1.1 / darknet format import.
"""

import argparse
import sys
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Version gate
# ---------------------------------------------------------------------------

def check_ultralytics():
    try:
        import ultralytics
    except ImportError:
        sys.exit("ultralytics not installed. Run: pip install -U ultralytics")

    from packaging.version import Version
    required = Version("8.4.0")
    installed = Version(ultralytics.__version__)
    if installed < required:
        sys.exit(
            f"ultralytics {installed} is too old; need >= {required}.\n"
            "Run: pip install -U ultralytics"
        )
    print(f"ultralytics {ultralytics.__version__}  ✓")

    # Confirm YOLOE exists
    try:
        from ultralytics import YOLOE  # noqa: F401
    except ImportError:
        sys.exit(
            "ultralytics.YOLOE not found in this build.\n"
            "Run: pip install -U ultralytics  and retry."
        )
    return ultralytics.__version__


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(p, base):
    p = Path(p)
    return p if p.is_absolute() else base / p


def to_yolo_line(box, img_w, img_h, class_id=0):
    """Convert xyxy box to YOLO normalised cx cy w h string."""
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 2: zero-shot pre-labelling")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold for pre-labels")
    parser.add_argument("--config", default="configs/train_person.yaml")
    parser.add_argument("--paths", default="configs/paths.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cfg = load_yaml(project_root / args.config)
    pcfg = load_yaml(project_root / args.paths)

    check_ultralytics()
    from ultralytics import YOLOE

    classes = cfg["classes"]
    frames_dir = resolve_path(pcfg["frames_dir"], project_root)
    prelabels_dir = resolve_path(pcfg["prelabels_dir"], project_root)
    manifest_path = frames_dir / "manifest.csv"

    if not manifest_path.exists():
        sys.exit(f"Manifest not found: {manifest_path}\nRun Step 1 first.")

    # ------------------------------------------------------------------
    # Collect frame paths from manifest
    # ------------------------------------------------------------------
    import csv
    frame_paths = []
    with open(manifest_path) as f:
        for row in csv.DictReader(f):
            p = Path(row["frame_path"])
            if p.exists():
                frame_paths.append(p)
            else:
                print(f"  [WARN] Frame not found, skipping: {p}")

    if not frame_paths:
        sys.exit("No valid frames found. Check manifest and frames_dir.")

    print(f"Frames to pre-label : {len(frame_paths)}")
    print(f"Classes             : {classes}")
    print(f"Confidence threshold: {args.conf}")

    # ------------------------------------------------------------------
    # Load YOLOE and set classes
    # ------------------------------------------------------------------
    print("\nLoading YOLOE-26s (zero-shot) ...")
    model = YOLOE("yoloe-26s-seg.pt")

    # set_classes tells the model which text prompts to detect
    model.set_classes(classes)

    # ------------------------------------------------------------------
    # Output structure mirrors the frames dir; also create classes.txt
    # ------------------------------------------------------------------
    images_out = prelabels_dir / "images"
    labels_out = prelabels_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    # Write classes.txt (YOLO 1.1 / darknet format)
    classes_file = prelabels_dir / "classes.txt"
    classes_file.write_text("\n".join(classes) + "\n")

    # CVAT / Label Studio import: obj.data (optional but helpful)
    obj_data = prelabels_dir / "obj.data"
    obj_data.write_text(
        f"classes = {len(classes)}\n"
        f"train  = obj_train_data.txt\n"
        f"names  = classes.txt\n"
        f"backup = backup/\n"
    )

    # ------------------------------------------------------------------
    # Run inference and write labels
    # ------------------------------------------------------------------
    import shutil
    import cv2

    total_boxes = 0
    empty_frames = 0

    for i, frame_path in enumerate(frame_paths):
        results = model.predict(
            source=str(frame_path),
            conf=args.conf,
            verbose=False,
        )
        r = results[0]
        img = cv2.imread(str(frame_path))
        h, w = img.shape[:2]

        # Symlink / copy image
        dest_img = images_out / frame_path.name
        if not dest_img.exists():
            shutil.copy2(frame_path, dest_img)

        # Write label file (empty if no detections)
        label_file = labels_out / (frame_path.stem + ".txt")
        lines = []

        boxes = r.boxes
        if boxes is not None and len(boxes) > 0:
            for box in boxes.xyxy.cpu().numpy():
                # Map detected class index back through model's class names
                # set_classes reorders names; class 0 == first in `classes`
                lines.append(to_yolo_line(box, w, h, class_id=0))

        label_file.write_text("\n".join(lines) + ("\n" if lines else ""))

        if not lines:
            empty_frames += 1
        else:
            total_boxes += len(lines)

        if (i + 1) % 100 == 0 or (i + 1) == len(frame_paths):
            print(f"  [{i+1}/{len(frame_paths)}] boxes so far: {total_boxes}  empty: {empty_frames}")

    # ------------------------------------------------------------------
    # Summary + import instructions
    # ------------------------------------------------------------------
    print("\n--- Summary ---")
    print(f"Frames processed  : {len(frame_paths)}")
    print(f"Boxes written     : {total_boxes}")
    print(f"Empty label files : {empty_frames}  "
          f"({100*empty_frames/len(frame_paths):.1f}% — target 10–20%)")
    print(f"Output directory  : {prelabels_dir}")
    print()
    print("=" * 60)
    print("IMPORT INSTRUCTIONS (YOLO 1.1 / darknet format)")
    print("=" * 60)
    print()
    print("CVAT (recommended):")
    print("  1. Create a new project. Add label 'person'.")
    print("  2. Create a task, upload images from:")
    print(f"       {images_out}")
    print("  3. In the task, click 'Upload annotations'.")
    print("     Format: YOLO 1.1")
    print(f"     Zip the contents of {labels_out} together with")
    print(f"     {classes_file} and upload the zip.")
    print()
    print("Label Studio:")
    print("  1. New project → Object Detection with Bounding Boxes.")
    print("  2. Import → YOLO → upload images folder AND labels folder.")
    print(f"     Image dir : {images_out}")
    print(f"     Label dir : {labels_out}")
    print(f"     Classes   : {classes_file}")
    print()
    print("LABELLING POLICY (see README.md for full policy):")
    print("  ✓  Box every real visible human, including partial people")
    print("     (edge of frame, shoulders only, arm entering frame).")
    print("  ✗  Do NOT box people in photos, posters, or on screens.")
    print("  ✓  Frames with no person: keep the label file empty")
    print("     (these are negative examples — target 10–20% of dataset).")


if __name__ == "__main__":
    main()
