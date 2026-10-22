"""
Step 7 — Speed benchmark on your GPU
Exports the best fine-tuned model to TensorRT FP16 (or ONNX fallback),
then benchmarks each model at batch=1, imgsz=640.

Metrics: median latency, p95 latency, FPS, peak VRAM, peak RAM.
Timing covers end-to-end: preprocess + inference + postprocess.

Usage:
    python scripts/benchmark.py [--config configs/train_person.yaml]
                                [--paths configs/paths.yaml]
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(p, base):
    p = Path(p)
    return p if p.is_absolute() else base / p


def reset_vram():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    gc.collect()


def peak_vram_mb() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1024 ** 2
    return 0.0


def current_ram_mb() -> float:
    proc = psutil.Process()
    return proc.memory_info().rss / 1024 ** 2


# ---------------------------------------------------------------------------
# Single model benchmark
# ---------------------------------------------------------------------------

def benchmark_model(model, warmup: int, n_frames: int, imgsz: int) -> dict:
    """Run end-to-end inference on random noise images; return latency stats."""
    import cv2

    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    dummy_path = None

    # Use a temp file so predict() receives a realistic input path
    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    cv2.imwrite(tmp.name, dummy)
    dummy_path = tmp.name
    tmp.close()

    reset_vram()
    ram_before = current_ram_mb()

    # Warm-up
    for _ in range(warmup):
        model.predict(source=dummy_path, imgsz=imgsz, verbose=False)

    reset_vram()

    # Timed runs
    latencies = []
    for _ in range(n_frames):
        t0 = time.perf_counter()
        model.predict(source=dummy_path, imgsz=imgsz, verbose=False)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)  # ms

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    vram_peak = peak_vram_mb()
    ram_peak  = current_ram_mb() - ram_before

    os.unlink(dummy_path)

    latencies = np.array(latencies)
    return {
        "median_ms":  float(np.median(latencies)),
        "p95_ms":     float(np.percentile(latencies, 95)),
        "fps":        1000.0 / float(np.median(latencies)),
        "vram_mb":    vram_peak,
        "ram_delta_mb": ram_peak,
        "n":          n_frames,
    }


# ---------------------------------------------------------------------------
# TensorRT export
# ---------------------------------------------------------------------------

def export_tensorrt(weights_path: Path, imgsz: int, half: bool) -> Path | None:
    from ultralytics import YOLOE
    try:
        print(f"  Exporting to TensorRT FP{'16' if half else '32'} ...")
        model = YOLOE(str(weights_path))
        exported = model.export(format="engine", imgsz=imgsz, half=half, device=0)
        engine_path = Path(exported) if isinstance(exported, str) else weights_path.with_suffix(".engine")
        if engine_path.exists():
            print(f"  TensorRT engine: {engine_path}")
            return engine_path
        else:
            print(f"  [WARN] Expected engine at {engine_path} — not found.")
            return None
    except Exception as e:
        print(f"  TensorRT export failed: {e}")
        return None


def export_onnx(weights_path: Path, imgsz: int) -> Path | None:
    from ultralytics import YOLOE
    try:
        print(f"  Falling back to ONNX export ...")
        model = YOLOE(str(weights_path))
        exported = model.export(format="onnx", imgsz=imgsz)
        onnx_path = Path(exported) if isinstance(exported, str) else weights_path.with_suffix(".onnx")
        if onnx_path.exists():
            print(f"  ONNX model: {onnx_path}")
            return onnx_path
        return None
    except Exception as e:
        print(f"  ONNX export also failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Step 7: speed benchmark")
    parser.add_argument("--config", default="configs/train_person.yaml")
    parser.add_argument("--paths",  default="configs/paths.yaml")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    cfg  = load_yaml(project_root / args.config)
    pcfg = load_yaml(project_root / args.paths)

    imgsz       = cfg["imgsz"]
    warmup      = cfg["warmup_frames"]
    n_frames    = cfg["benchmark_frames"]
    trt_half    = cfg["trt_half"]
    prod_weights = pcfg.get("production_weights")
    classes      = cfg["classes"]

    weights_registry = project_root / "runs" / "finetuned_weights.yaml"
    reports_dir = resolve_path(pcfg["reports_dir"], project_root)
    reports_dir.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLOE, YOLO

    print(f"torch      : {torch.__version__}")
    print(f"CUDA       : {torch.version.cuda}")
    print(f"GPU        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"imgsz      : {imgsz}")
    print(f"warmup     : {warmup} frames")
    print(f"benchmark  : {n_frames} frames")

    # ------------------------------------------------------------------
    # Build list of (name, pt_path) pairs to benchmark
    # ------------------------------------------------------------------
    to_benchmark: list[tuple[str, Path]] = []

    # Zero-shot YOLOE (PyTorch only)
    to_benchmark.append(("YOLOE-26s zero-shot (PyTorch)", None))

    # COCO YOLO
    to_benchmark.append(("COCO YOLO pretrained (PyTorch)", None))

    # Production model
    if prod_weights and Path(prod_weights).exists():
        to_benchmark.append(("Production model (PyTorch)", Path(prod_weights)))

    # Fine-tuned models
    if weights_registry.exists():
        fw = load_yaml(weights_registry)
        if fw.get("full") and Path(fw["full"]).exists():
            to_benchmark.append(("YOLOE fine-tuned full (PyTorch)", Path(fw["full"])))
        if fw.get("frozen") and Path(fw["frozen"]).exists():
            to_benchmark.append(("YOLOE fine-tuned frozen (PyTorch)", Path(fw["frozen"])))
    else:
        print("[WARN] finetuned_weights.yaml not found — fine-tuned benchmarks skipped.")

    # ------------------------------------------------------------------
    # Export best fine-tuned model to TensorRT / ONNX
    # ------------------------------------------------------------------
    best_ft_path: Path | None = None
    if weights_registry.exists():
        fw = load_yaml(weights_registry)
        best_ft_path = Path(fw.get("full") or fw.get("frozen") or "")
        if not (best_ft_path and best_ft_path.exists()):
            best_ft_path = None

    exported_path: Path | None = None
    export_format = None

    if best_ft_path:
        print(f"\nExporting {best_ft_path.name} ...")
        exported_path = export_tensorrt(best_ft_path, imgsz, trt_half)
        if exported_path:
            export_format = f"TensorRT FP{'16' if trt_half else '32'}"
        else:
            exported_path = export_onnx(best_ft_path, imgsz)
            if exported_path:
                export_format = "ONNX"
                print("  NOTE: TensorRT unavailable; using ONNX. For TensorRT, install tensorrt and torch2trt.")

        if exported_path:
            to_benchmark.append((f"YOLOE fine-tuned full ({export_format})", exported_path))

    # ------------------------------------------------------------------
    # Benchmark
    # ------------------------------------------------------------------
    results = {}

    for name, pt_path in to_benchmark:
        print(f"\n--- {name} ---")
        try:
            if pt_path is None:
                if "zero-shot" in name:
                    model = YOLOE("yoloe-26s-seg.pt")
                    model.set_classes(classes)
                else:
                    try:
                        model = YOLO("yolo11n.pt")
                    except Exception:
                        model = YOLO("yolov8n.pt")
            elif pt_path.suffix == ".engine":
                model = YOLOE(str(pt_path))
            elif pt_path.suffix == ".onnx":
                model = YOLOE(str(pt_path))
            else:
                model = YOLOE(str(pt_path))

            stats = benchmark_model(model, warmup, n_frames, imgsz)
            results[name] = stats
            print(f"  median={stats['median_ms']:.1f}ms  p95={stats['p95_ms']:.1f}ms  "
                  f"FPS={stats['fps']:.1f}  VRAM={stats['vram_mb']:.0f}MB  "
                  f"RAM+{stats['ram_delta_mb']:.0f}MB")
        except Exception as e:
            print(f"  FAILED: {e}")
            results[name] = {"error": str(e)}

    # ------------------------------------------------------------------
    # Print and save summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print(f"BENCHMARK RESULTS  (batch=1, imgsz={imgsz}, warmup={warmup}, n={n_frames})")
    print("=" * 90)
    fmt = "{:<45} {:>10} {:>8} {:>8} {:>10} {:>12}"
    print(fmt.format("Model", "median(ms)", "p95(ms)", "FPS", "VRAM(MB)", "RAM_delta"))
    print("-" * 90)
    for name, s in results.items():
        if "error" in s:
            print(fmt.format(name[:45], "ERR", "ERR", "ERR", "ERR", s["error"][:12]))
        else:
            print(fmt.format(
                name[:45],
                f"{s['median_ms']:.1f}",
                f"{s['p95_ms']:.1f}",
                f"{s['fps']:.1f}",
                f"{s['vram_mb']:.0f}",
                f"{s['ram_delta_mb']:.0f}",
            ))

    bench_out = reports_dir / "benchmark_results.yaml"
    with open(bench_out, "w") as f:
        yaml.dump(results, f)
    print(f"\nResults saved to: {bench_out}")

    if export_format:
        print(f"\nExported model format : {export_format}")
    else:
        print("\nNo export produced (fine-tuned weights not found or export failed).")


if __name__ == "__main__":
    main()
