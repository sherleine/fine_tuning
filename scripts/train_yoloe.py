"""
Step 5 — Fine-tune YOLOE-26s with YOLOEPETrainer
Runs two configurations:
  (a) full fine-tune
  (b) partially frozen backbone

Usage:
    python scripts/train_yoloe.py [--run {full,frozen,both}]
                                  [--config configs/train_person.yaml]
                                  [--paths configs/paths.yaml]

Logs ultralytics, torch, and CUDA versions. All hyperparameters come from
configs/train_person.yaml — no hard-coded values here.
"""

import argparse
import sys
import time
from pathlib import Path

import yaml


# ---------------------------------------------------------------------------
# Version logging
# ---------------------------------------------------------------------------

def log_versions():
    import torch
    try:
        import ultralytics
        ul_ver = ultralytics.__version__
    except Exception:
        ul_ver = "unknown"

    cuda_ver = torch.version.cuda or "N/A"
    device   = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"

    print("=" * 60)
    print("Environment")
    print(f"  ultralytics : {ul_ver}")
    print(f"  torch       : {torch.__version__}")
    print(f"  CUDA        : {cuda_ver}")
    print(f"  device      : {device}  ({gpu_name})")
    print("=" * 60)

    return {"ultralytics": ul_ver, "torch": torch.__version__, "cuda": cuda_ver}


# ---------------------------------------------------------------------------
# Backbone inspection (run with inspect_backbone: true in config)
# ---------------------------------------------------------------------------

def inspect_backbone(model):
    """Print model layer names so the user can choose freeze_n."""
    print("\n--- Model layers (index : name) ---")
    for i, (name, _) in enumerate(model.model.named_children()):
        print(f"  {i:3d} : {name}")
    print()
    print("Set `freeze_n` in configs/train_person.yaml to the index of the")
    print("first NON-backbone layer (typically the neck / head start).")
    print("Exiting (inspect_backbone: true).")
    sys.exit(0)


# ---------------------------------------------------------------------------
# Single training run
# ---------------------------------------------------------------------------

def run_training(
    model,
    cfg: dict,
    dataset_yaml: Path,
    run_name: str,
    runs_dir: Path,
    freeze: int | None,
):
    import torch
    from ultralytics.models.yolo.yoloe import YOLOEPETrainer

    epochs      = cfg["epochs"]
    patience    = cfg["patience"]
    imgsz       = cfg["imgsz"]
    batch       = cfg["batch"]
    batch_fb    = cfg["batch_fallback"]
    imgsz_fb    = cfg["imgsz_fallback"]
    seed        = cfg["seed"]
    workers     = cfg["workers"]

    train_kwargs = dict(
        data     = str(dataset_yaml),
        epochs   = epochs,
        patience = patience,
        imgsz    = imgsz,
        batch    = batch,
        seed     = seed,
        workers  = workers,
        project  = str(runs_dir),
        name     = run_name,
        exist_ok = False,
        # Hyperparameters from config
        optimizer    = cfg["optimizer"],
        lr0          = cfg["lr0"],
        lrf          = cfg["lrf"],
        momentum     = cfg["momentum"],
        weight_decay = cfg["weight_decay"],
        warmup_epochs     = cfg["warmup_epochs"],
        warmup_momentum   = cfg["warmup_momentum"],
        warmup_bias_lr    = cfg["warmup_bias_lr"],
        close_mosaic      = cfg["close_mosaic"],
        hsv_h       = cfg["hsv_h"],
        hsv_s       = cfg["hsv_s"],
        hsv_v       = cfg["hsv_v"],
        degrees     = cfg["degrees"],
        translate   = cfg["translate"],
        scale       = cfg["scale"],
        shear       = cfg["shear"],
        perspective = cfg["perspective"],
        flipud      = cfg["flipud"],
        fliplr      = cfg["fliplr"],
        mosaic      = cfg["mosaic"],
        mixup       = cfg["mixup"],
        trainer     = YOLOEPETrainer,
    )

    if freeze is not None:
        train_kwargs["freeze"] = freeze

    # First attempt
    try:
        print(f"\nStarting training run: {run_name}")
        print(f"  epochs={epochs}, imgsz={imgsz}, batch={batch}, freeze={freeze}")
        t0 = time.time()
        model.train(**train_kwargs)
        elapsed = time.time() - t0
        print(f"Training finished in {elapsed/60:.1f} min")

    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        if "out of memory" in str(e).lower() or isinstance(e, torch.cuda.OutOfMemoryError):
            print(f"[OOM] Retrying with batch={batch_fb} ...")
            torch.cuda.empty_cache()
            train_kwargs["batch"] = batch_fb
            try:
                model.train(**train_kwargs)
            except (RuntimeError, torch.cuda.OutOfMemoryError) as e2:
                if "out of memory" in str(e2).lower() or isinstance(e2, torch.cuda.OutOfMemoryError):
                    print(f"[OOM] Retrying with batch={batch_fb}, imgsz={imgsz_fb} ...")
                    torch.cuda.empty_cache()
                    train_kwargs["imgsz"] = imgsz_fb
                    model.train(**train_kwargs)
                else:
                    raise
        else:
            raise

    # Return path to best weights
    best = runs_dir / run_name / "weights" / "best.pt"
    if not best.exists():
        # ultralytics sometimes nests differently
        candidates = list((runs_dir / run_name).glob("**/best.pt"))
        best = candidates[0] if candidates else None

    return best


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 5: fine-tune YOLOE-26s")
    parser.add_argument("--run", choices=["full", "frozen", "both"], default="both",
                        help="Which training run(s) to execute")
    parser.add_argument("--config", default="configs/train_person.yaml")
    parser.add_argument("--paths",  default="configs/paths.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cfg  = yaml.safe_load(open(project_root / args.config))
    pcfg = yaml.safe_load(open(project_root / args.paths))

    env_info = log_versions()

    # Check YOLOEPETrainer is available
    try:
        from ultralytics.models.yolo.yoloe import YOLOEPETrainer
        print("YOLOEPETrainer found  ✓")
    except ImportError:
        sys.exit(
            "YOLOEPETrainer not found in ultralytics.\n"
            "Run: pip install -U ultralytics  (need >= 8.4.0)\n"
            "If still missing, check: python -c \"import ultralytics; print(ultralytics.__version__)\""
        )

    from ultralytics import YOLOE

    dataset_dir  = Path(pcfg["dataset_dir"]) if Path(pcfg["dataset_dir"]).is_absolute() else project_root / pcfg["dataset_dir"]
    dataset_yaml = dataset_dir / f"{dataset_dir.name}.yaml"
    runs_dir     = Path(pcfg["runs_dir"])   if Path(pcfg["runs_dir"]).is_absolute()   else project_root / pcfg["runs_dir"]

    if not dataset_yaml.exists():
        sys.exit(f"Dataset YAML not found: {dataset_yaml}\nRun Step 3 first.")

    runs_dir.mkdir(parents=True, exist_ok=True)

    best_weights = {}

    # ------------------------------------------------------------------
    # Backbone inspection mode (set inspect_backbone: true in config)
    # ------------------------------------------------------------------
    if cfg.get("inspect_backbone", False):
        model = YOLOE("yoloe-26s-seg.pt")
        inspect_backbone(model)

    # ------------------------------------------------------------------
    # (a) Full fine-tune
    # ------------------------------------------------------------------
    if args.run in ("full", "both"):
        model_full = YOLOE("yoloe-26s-seg.pt")
        best = run_training(
            model  = model_full,
            cfg    = cfg,
            dataset_yaml = dataset_yaml,
            run_name = "yoloe26s_full",
            runs_dir = runs_dir,
            freeze   = None,
        )
        best_weights["full"] = str(best) if best else None
        print(f"\n  Full fine-tune best weights: {best}")

    # ------------------------------------------------------------------
    # (b) Partially frozen backbone
    # ------------------------------------------------------------------
    if args.run in ("frozen", "both"):
        freeze_n = cfg.get("freeze_n", 10)
        print(f"\n  Freeze setting: freeze_n={freeze_n}")
        print(f"  Layers 0..{freeze_n-1} (backbone) will be frozen.")
        print(f"  Rationale: YOLOE-26s backbone typically ends around layer 9–10")
        print(f"  (P3/P4/P5 CSP layers). Freezing them lets the neck+head adapt")
        print(f"  faster while preserving pretrained visual features. Adjust")
        print(f"  freeze_n in configs/train_person.yaml after inspecting layers")
        print(f"  with inspect_backbone: true.")

        model_frozen = YOLOE("yoloe-26s-seg.pt")
        best = run_training(
            model  = model_frozen,
            cfg    = cfg,
            dataset_yaml = dataset_yaml,
            run_name = f"yoloe26s_frozen{freeze_n}",
            runs_dir = runs_dir,
            freeze   = freeze_n,
        )
        best_weights["frozen"] = str(best) if best else None
        print(f"\n  Frozen fine-tune best weights: {best}")

    # ------------------------------------------------------------------
    # Save weight paths for downstream scripts
    # ------------------------------------------------------------------
    weights_registry = project_root / "runs" / "finetuned_weights.yaml"
    with open(weights_registry, "w") as f:
        yaml.dump(best_weights, f)

    print("\n--- Training complete ---")
    for tag, path in best_weights.items():
        print(f"  {tag:10s} : {path}")
    print(f"\nWeight paths saved to: {weights_registry}")
    print("Next: run scripts/evaluate.py to compare all models on the test set.")


if __name__ == "__main__":
    main()
