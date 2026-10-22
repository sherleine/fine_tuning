# YOLOE-26s Person-Detection Fine-Tuning Pilot

Fine-tunes Ultralytics YOLOE-26s on your own proctoring footage for the
single class **person**, with a full evaluation pipeline comparing it against
zero-shot YOLOE, COCO-pretrained YOLO, and your current production model.

---

## Prerequisites

```bash
pip install -U ultralytics>=8.4.0
pip install imagehash Pillow opencv-python pyyaml psutil
# Optional but recommended for TensorRT export:
# pip install tensorrt  (see https://docs.ultralytics.com/integrations/tensorrt)
```

Verify the key API is present:

```python
python -c "from ultralytics import YOLOE; from ultralytics.models.yolo.yoloe import YOLOEPETrainer; print('OK')"
```

---

## Configuration — do this first

### 1. `configs/paths.yaml`

Edit these two fields at minimum:

| Key | What to set |
|-----|-------------|
| `videos_dir` | Absolute path to your raw session videos |
| `video_regex` | Python regex with named groups `session_id` and `camera` |
| `production_weights` | Path to your current production `.pt` file (or `null` to skip) |

**Example `video_regex`** for files like `sess42_primary_20241015T090000.mp4`:
```yaml
video_regex: "sess(?P<session_id>[^_]+)_(?P<camera>primary|secondary)_.*\\.mp4"
```

### 2. `configs/train_person.yaml`

Defaults are tuned for an RTX 3050 6 GB. Key knobs:

| Key | Default | Notes |
|-----|---------|-------|
| `epochs` | 80 | Increase for larger datasets |
| `batch` | -1 | Auto-batch; set to 8 if it OOMs |
| `freeze_n` | 10 | Layers 0–9 frozen in the partial-freeze run |
| `inspect_backbone` | false | Set `true` to print layer names and exit |

---

## Command sequence (Steps 1–7)

Run each command from the project root (`d:\fine tuning for people`).

### Step 1 — Extract frames

```bash
python scripts/extract_frames.py --fps 1 --hash-threshold 8 --max-per-session 500
```

**Review output:** Check `data/frames/manifest.csv` and spot-check a few frames.
Adjust `--fps` or `--max-per-session` if you want more/fewer frames.

---

### Step 2 — Pre-labelling (zero-shot YOLOE)

```bash
python scripts/prelabel.py --conf 0.25
```

**⚠ STOP — manual label correction required before proceeding.**

Import `data/prelabels/` into CVAT or Label Studio (instructions printed by the script).

**Labelling policy:**
- ✅ Box **every real, visible human**, including partial people (edge of frame, shoulders only, arm entering).
- ❌ Do **NOT** box people in photos, posters, or on screens.
- ✅ Frames with **no person** → keep the label file empty (these are negatives; target 10–20% of the dataset).

After correcting all labels, **overwrite the files in `data/prelabels/labels/`** with the corrected versions (CVAT/Label Studio can export back to YOLO format).

---

### Step 3 — Dataset split

```bash
python scripts/split_dataset.py --train 0.70 --val 0.15 --seed 42
```

Splits by session ID so no session appears in two splits.
Prints per-split counts and runs a leakage check.

---

### Step 4 — Baseline evaluation

```bash
python scripts/eval_baselines.py
```

Evaluates on the TEST split:
1. Zero-shot YOLOE-26s
2. COCO-pretrained YOLO
3. Your production model (if configured)

Results saved to `reports/baselines.yaml`.

---

### Step 5 — Fine-tuning

```bash
# Both runs (recommended)
python scripts/train_yoloe.py --run both

# Or individually:
python scripts/train_yoloe.py --run full
python scripts/train_yoloe.py --run frozen
```

If you want to inspect model layers before choosing `freeze_n`:

```bash
# In configs/train_person.yaml, temporarily set:  inspect_backbone: true
python scripts/train_yoloe.py --run frozen
# Then set inspect_backbone: false and choose freeze_n
```

Weights saved under `runs/yoloe26s_full/` and `runs/yoloe26s_frozenN/`.

---

### Step 6 — Full evaluation and report

```bash
python scripts/evaluate.py
```

- Tunes confidence threshold on VAL (maximise F1, subject to precision ≥ 0.90).
- Evaluates all models on TEST.
- Per-slice recall: primary vs secondary camera, multi-person, small/partial people, low-light.
- False positives per 1,000 empty frames.
- Person-count accuracy per frame.
- Saves `reports/person_pilot.md` + annotated FP/FN images.

---

### Step 7 — Speed benchmark

```bash
python scripts/benchmark.py
```

- Exports best fine-tuned model to TensorRT FP16 (falls back to ONNX if unavailable).
- Benchmarks: 50 warm-up + 1,000 timed frames at batch=1, imgsz=640.
- Reports median/p95 latency, FPS, peak VRAM, peak RAM.
- Results saved to `reports/benchmark_results.yaml`.

---

## Output structure

```
runs/
  yoloe26s_full/weights/best.pt
  yoloe26s_frozenN/weights/best.pt
  finetuned_weights.yaml           ← paths used by Steps 6 & 7

reports/
  baselines.yaml
  benchmark_results.yaml
  person_pilot.md                  ← final comparison report
  fp_images/                       ← 20 worst false positives (annotated)
  fn_images/                       ← 20 worst false negatives (annotated)

data/
  frames/                          ← Step 1 output
    manifest.csv
    <session_id>/<frame>.jpg
  prelabels/                       ← Step 2 output (correct here!)
    images/
    labels/
    classes.txt
  person/                          ← Step 3 output (final dataset)
    images/{train,val,test}/
    labels/{train,val,test}/
    person.yaml
```

---

## Adding more classes later

1. Add class names to `configs/train_person.yaml` under `classes:`.
2. Re-run Steps 2–7 (label the new class in CVAT, then split/train/eval).
3. No code changes needed — all scripts read the class list from config.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `YOLOE not found` | `pip install -U ultralytics` |
| `YOLOEPETrainer not found` | Same; need ultralytics >= 8.4.0 |
| OOM during training | Set `batch: 8` or `imgsz_fallback: 512` in `train_person.yaml` |
| `model.val()` fails (seg/detect mismatch) | Script auto-falls back to predict+manual metrics |
| TensorRT export fails | Check `pip install tensorrt`; script falls back to ONNX |
| Empty `video_regex` matches | Test your regex: `python -c "import re; print(re.search('<your_regex>', '<your_filename>'))"` |
