"""
Convert the downloaded Open Images v7 fiftyone dataset to YOLO format
ready for fine-tuning.

Usage:
    python scripts/convert_openimages.py

Output:
    data/openimages_person/
        images/train/
        images/val/
        labels/train/
        labels/val/
        openimages_person.yaml
"""

import shutil
import sys
from pathlib import Path

import yaml


def main():
    try:
        import fiftyone as fo
    except ImportError:
        sys.exit("fiftyone not installed. Run: pip install fiftyone")

    project_root = Path(__file__).resolve().parent.parent
    out_dir = project_root / "data" / "openimages_person"

    # ------------------------------------------------------------------
    # Load the dataset fiftyone downloaded
    # ------------------------------------------------------------------
    dataset_name = "open-images-v7-train-5000"
    print(f"Loading fiftyone dataset: {dataset_name}")

    try:
        ds = fo.load_dataset(dataset_name)
    except Exception:
        # Try listing available datasets
        available = fo.list_datasets()
        oi_datasets = [d for d in available if "open-images" in d.lower()]
        if not oi_datasets:
            sys.exit(
                "Open Images dataset not found in fiftyone.\n"
                "Re-run the download command:\n"
                "  python -c \"import fiftyone.zoo as foz; "
                "foz.load_zoo_dataset('open-images-v7', split='train', "
                "label_types=['detections'], classes=['Person'], max_samples=5000)\""
            )
        dataset_name = oi_datasets[-1]
        print(f"  Found: {dataset_name}")
        ds = fo.load_dataset(dataset_name)

    print(f"Total samples: {len(ds)}")

    # ------------------------------------------------------------------
    # Filter to samples that actually have Person detections
    # ------------------------------------------------------------------
    ds_person = ds.filter_labels(
        "ground_truth",
        fo.ViewField("label") == "Person"
    )
    print(f"Samples with Person label: {len(ds_person)}")

    # ------------------------------------------------------------------
    # 80/20 split for train/val
    # ------------------------------------------------------------------
    total   = len(ds_person)
    n_train = int(total * 0.8)
    samples = list(ds_person)

    import random
    random.seed(42)
    random.shuffle(samples)

    splits = {
        "train": samples[:n_train],
        "val":   samples[n_train:],
    }

    # ------------------------------------------------------------------
    # Write YOLO format
    # ------------------------------------------------------------------
    stats = {}
    for split_name, split_samples in splits.items():
        img_out = out_dir / "images" / split_name
        lbl_out = out_dir / "labels" / split_name
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        n_images = n_boxes = n_empty = 0

        for sample in split_samples:
            src = Path(sample.filepath)
            if not src.exists():
                continue

            # Copy image
            dst_img = img_out / src.name
            shutil.copy2(src, dst_img)

            # Write label
            lbl_file = lbl_out / (src.stem + ".txt")
            lines = []

            if sample.ground_truth and sample.ground_truth.detections:
                for det in sample.ground_truth.detections:
                    if det.label != "Person":
                        continue
                    # fiftyone stores [x, y, w, h] normalised (top-left origin)
                    x, y, w, h = det.bounding_box
                    cx = x + w / 2
                    cy = y + h / 2
                    # clamp to [0, 1]
                    cx = max(0.0, min(1.0, cx))
                    cy = max(0.0, min(1.0, cy))
                    w  = max(0.0, min(1.0, w))
                    h  = max(0.0, min(1.0, h))
                    lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            lbl_file.write_text("\n".join(lines) + ("\n" if lines else ""))
            n_images += 1
            n_boxes  += len(lines)
            if not lines:
                n_empty += 1

        stats[split_name] = {"images": n_images, "boxes": n_boxes, "empty": n_empty}

    # ------------------------------------------------------------------
    # Write dataset YAML
    # ------------------------------------------------------------------
    yaml_path = out_dir / "openimages_person.yaml"
    dataset_yaml = {
        "path":  str(out_dir.resolve()),
        "train": "images/train",
        "val":   "images/val",
        "nc":    1,
        "names": {0: "person"},
    }
    with open(yaml_path, "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False, sort_keys=False)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("Open Images → YOLO conversion complete")
    print("=" * 55)
    for split, s in stats.items():
        print(f"  {split:<6}: {s['images']:>5} images  "
              f"{s['boxes']:>6} boxes  "
              f"{s['empty']:>4} empty")
    print(f"\nDataset YAML: {yaml_path}")
    print()
    print("Next — fine-tune on this dataset:")
    print(f"  python scripts/finetune_compare.py --data {yaml_path}")


if __name__ == "__main__":
    main()
