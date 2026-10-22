"""
Step 3 — Dataset split by session ID (no frame leakage between splits)
Reads corrected labels from the prelabels directory and splits into
train/val/test, ensuring all frames from a session stay in one split.

Usage:
    python scripts/split_dataset.py [--train 0.70] [--val 0.15]
                                    [--seed 42]
                                    [--config configs/train_person.yaml]
                                    [--paths configs/paths.yaml]

Output:
    data/person/images/{train,val,test}/
    data/person/labels/{train,val,test}/
    data/person/person.yaml
"""

import argparse
import csv
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import yaml


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(p, base):
    p = Path(p)
    return p if p.is_absolute() else base / p


def count_boxes(label_path: Path) -> int:
    text = label_path.read_text().strip()
    return len(text.splitlines()) if text else 0


def frame_short_side(label_path: Path) -> float | None:
    """Return width*height of the bounding box as a fraction of image area."""
    # We can't open the image here easily; we approximate from label coords.
    # Returns the minimum box area fraction across all boxes (for "small people" slice).
    lines = label_path.read_text().strip().splitlines()
    if not lines:
        return None
    areas = []
    for line in lines:
        parts = line.split()
        if len(parts) == 5:
            bw, bh = float(parts[3]), float(parts[4])
            areas.append(bw * bh)
    return min(areas) if areas else None


def main():
    parser = argparse.ArgumentParser(description="Step 3: session-safe dataset split")
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val",   type=float, default=0.15)
    parser.add_argument("--seed",  type=int,   default=42)
    parser.add_argument("--config", default="configs/train_person.yaml")
    parser.add_argument("--paths",  default="configs/paths.yaml")
    args = parser.parse_args()

    test_frac = 1.0 - args.train - args.val
    if test_frac < 0.01:
        sys.exit("--train + --val must be < 1.0")

    project_root = Path(__file__).resolve().parent.parent
    cfg = load_yaml(project_root / args.config)
    pcfg = load_yaml(project_root / args.paths)

    classes = cfg["classes"]
    prelabels_dir = resolve_path(pcfg["prelabels_dir"], project_root)
    dataset_dir   = resolve_path(pcfg["dataset_dir"],   project_root)
    manifest_path = resolve_path(pcfg["frames_dir"], project_root) / "manifest.csv"

    images_src = prelabels_dir / "images"
    labels_src = prelabels_dir / "labels"

    if not images_src.exists() or not labels_src.exists():
        sys.exit(
            f"Pre-labelled data not found at {prelabels_dir}\n"
            "Run Step 2 (prelabel.py) and correct labels first."
        )

    # ------------------------------------------------------------------
    # Load manifest to know session IDs
    # ------------------------------------------------------------------
    frame_to_session: dict[str, str] = {}
    frame_to_camera: dict[str, str] = {}

    if manifest_path.exists():
        with open(manifest_path) as f:
            for row in csv.DictReader(f):
                stem = Path(row["frame_path"]).stem
                frame_to_session[stem] = row["session_id"]
                frame_to_camera[stem]  = row["camera"]
    else:
        # Fall back: parse session_id from filename stem (first token before _)
        print("[WARN] manifest.csv not found; inferring session_id from filename prefix.")

    # ------------------------------------------------------------------
    # Collect all frames that have both an image and a label file
    # ------------------------------------------------------------------
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_img_paths = sorted(
        p for p in images_src.iterdir() if p.suffix.lower() in img_exts
    )

    valid_frames = []
    for img_path in all_img_paths:
        lbl_path = labels_src / (img_path.stem + ".txt")
        if not lbl_path.exists():
            print(f"  [WARN] No label for {img_path.name} — skipping")
            continue
        valid_frames.append((img_path, lbl_path))

    if not valid_frames:
        sys.exit("No valid image+label pairs found.")

    print(f"Valid frame pairs: {len(valid_frames)}")

    # ------------------------------------------------------------------
    # Group frames by session ID
    # ------------------------------------------------------------------
    session_to_frames: dict[str, list] = defaultdict(list)
    for img_path, lbl_path in valid_frames:
        stem = img_path.stem
        if stem in frame_to_session:
            sid = frame_to_session[stem]
        else:
            # Fallback: first underscore-separated token
            sid = stem.split("_")[0]
        session_to_frames[sid].append((img_path, lbl_path))

    sessions = sorted(session_to_frames.keys())
    print(f"Sessions found    : {len(sessions)}")

    # ------------------------------------------------------------------
    # Split sessions (not frames) into train/val/test
    # ------------------------------------------------------------------
    random.seed(args.seed)
    random.shuffle(sessions)

    n = len(sessions)
    n_train = max(1, round(n * args.train))
    n_val   = max(1, round(n * args.val))
    # test gets whatever is left (at least 1 if there are 3+ sessions)
    if n < 3:
        sys.exit(f"Need at least 3 sessions to create 3 splits, got {n}.")

    train_sessions = set(sessions[:n_train])
    val_sessions   = set(sessions[n_train:n_train + n_val])
    test_sessions  = set(sessions[n_train + n_val:])

    # Safety check: no session in more than one split
    overlap_tv = train_sessions & val_sessions
    overlap_tt = train_sessions & test_sessions
    overlap_vt = val_sessions   & test_sessions
    if overlap_tv or overlap_tt or overlap_vt:
        sys.exit(
            f"BUG: sessions appear in multiple splits!\n"
            f"  train∩val={overlap_tv}\n"
            f"  train∩test={overlap_tt}\n"
            f"  val∩test={overlap_vt}"
        )

    split_map = (
        [("train", train_sessions)]
        + [("val",   val_sessions)]
        + [("test",  test_sessions)]
    )

    # ------------------------------------------------------------------
    # Copy files into data/person/{images,labels}/{split}/
    # ------------------------------------------------------------------
    stats: dict[str, dict] = {}

    for split_name, split_sessions in split_map:
        img_out = dataset_dir / "images" / split_name
        lbl_out = dataset_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        n_images = 0
        n_boxes  = 0
        n_empty  = 0
        n_multi  = 0
        cam_counts: dict[str, int] = defaultdict(int)

        for sid in split_sessions:
            for img_path, lbl_path in session_to_frames[sid]:
                shutil.copy2(img_path, img_out / img_path.name)
                shutil.copy2(lbl_path, lbl_out / lbl_path.name)

                boxes = count_boxes(lbl_path)
                n_images += 1
                n_boxes  += boxes
                if boxes == 0:
                    n_empty += 1
                if boxes >= 2:
                    n_multi += 1

                stem = img_path.stem
                cam = frame_to_camera.get(stem, "unknown")
                cam_counts[cam] += 1

        stats[split_name] = {
            "sessions": len(split_sessions),
            "images":   n_images,
            "boxes":    n_boxes,
            "empty":    n_empty,
            "multi":    n_multi,
            "cameras":  dict(cam_counts),
        }

    # ------------------------------------------------------------------
    # Write dataset YAML
    # ------------------------------------------------------------------
    yaml_path = dataset_dir / f"{dataset_dir.name}.yaml"
    dataset_yaml = {
        "path":  str(dataset_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "test":  "images/test",
        "nc":    len(classes),
        "names": {i: name for i, name in enumerate(classes)},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # Print summary
    # ------------------------------------------------------------------
    print("\n--- Split Summary ---")
    print(f"{'Split':<8} {'Sessions':>8} {'Images':>8} {'Boxes':>8} {'Empty':>8} {'Multi':>8}  Cameras")
    print("-" * 70)
    for split_name, s in stats.items():
        cam_str = ", ".join(f"{k}:{v}" for k, v in sorted(s["cameras"].items()))
        print(
            f"{split_name:<8} {s['sessions']:>8} {s['images']:>8} "
            f"{s['boxes']:>8} {s['empty']:>8} {s['multi']:>8}  {cam_str}"
        )

    print(f"\nDataset YAML written to: {yaml_path}")
    print(f"Session integrity check : PASSED (no leakage between splits)")

    # Final sanity: re-verify no frame appears in two splits
    seen = set()
    for split_name, _ in split_map:
        img_out = dataset_dir / "images" / split_name
        for p in img_out.iterdir():
            if p.name in seen:
                sys.exit(f"INTEGRITY ERROR: {p.name} appears in multiple splits!")
            seen.add(p.name)
    print("Frame uniqueness check  : PASSED")


if __name__ == "__main__":
    main()
