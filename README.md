# GPU telemetry workload classification

Can ordinary NVIDIA management telemetry show what a GPU is doing (training,
inference or non-ML work), and which LLM is being served? This repo covers the
whole loop: run labelled workloads, record 1 Hz NVML telemetry, and train
classifiers on 30 s windows of it.

Merged from `hap-telemetry`, `gpu-live-dashboard` and `gpu-telemetry-probe`.

## Layout

```
run_workloads.py       1. start workloads: the run matrix, dispatched across GPUs
collect_telemetry.py   2. collect telemetry: 1 Hz NVML -> labelled parquet trace
train_classifier.py    3. train/evaluate classifiers; score traces with a bundle
gputel/
  workloads.py         the workloads themselves (vision, gradprobe, llm, idle, fft, ...)
  features.py          loading, labelling, windowing, 159 window features
  nvml.py              NVML reader shared by the collector and the dashboard
dashboard/             live classifier dashboard (python -m dashboard)
docs/INTERPRETING.md   how to read the dashboard, and how it can mislead
data/                  your traces (gitignored; see below)
```

## Data

No telemetry traces are included in this repo. Collect your own with steps 1
and 2 below; `data/` is gitignored so traces stay local. The examples assume
one directory per power regime and task, e.g. `data/workload`,
`data/model_id` and `data/batch_sweep`.

## Install

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Training only needs the first block of `requirements.txt`. Collection and
workloads need an NVIDIA GPU, `nvidia-ml-py` and PyTorch. The dashboard needs
`fastapi` and `uvicorn`, plus `nvidia-ml-py` for live mode.

## 1. Start workloads

```bash
python run_workloads.py --list                              # the matrix and each label's class
python run_workloads.py --gpus 0,1 --dry-run
python run_workloads.py --gpus 0,1 --duration 600 --reps 3 --output-dir data/new
python run_workloads.py --gpus 0 --only llm_infer_sweep     # filter by label substring
```

Each run is recorded by `collect_telemetry.py`, so each run writes one trace.
Progress is saved to `<output-dir>/progress.json` after every run, so the same
command resumes after an interruption (`--fresh` starts over). Workload output
goes to `<output-dir>/logs/`. Before each run the GPU is checked: a GPU that
another process is using is waited on, and one that fails a CUDA kernel test
is retired and its task handed to another GPU. GPUs are never enumerated
automatically, so pass only the ones you're allowed to use.

## 2. Collect telemetry

```bash
python collect_telemetry.py --gpu 0 --label idle --duration 600
python collect_telemetry.py --gpu 0 --label my_train_job -- python train.py
```

This records nine signals (utilisation, memory utilisation, memory used,
power, temperature, SM and memory clocks, PCIe TX and RX) plus encoder,
decoder and fan data. A wrapped command runs pinned to the same GPU (with
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, so CUDA and NVML use the same index) after a
2 s idle baseline. Recording stops when the command exits.

**Telemetry covers the whole GPU.** Another user's job on the same device gets
recorded under your label. The collector warns if the GPU is busy when it
starts.

## 3. Train a classifier

```bash
python train_classifier.py workload --data data/workload --out models/workload
python train_classifier.py model_id --data data/model_id --out models/model_id
python train_classifier.py model_id --data data/batch_sweep --out models/sweep
python train_classifier.py score --bundle models/workload/threeway.joblib data/new/*.parquet
```

Each output directory gets `.joblib` bundles (model, exact feature column
order, classes, window settings), `metadata.json` (corpus, provenance, run IDs
used for training, full CV scores and confusion matrices) and `report.md`.

- **workload** trains `binary` (training vs rest) and `threeway`
  (`ml_training` / `ml_inference` / `other`).
- **model_id** trains with and without memory-derived features, plus a control
  that uses only `mem_used_mb_mean`. On a batch sweep it also holds out one
  batch size at a time.
- `--holdout-config LABEL` (repeatable) leaves configs out of training
  entirely. Scoring those configs later is then truly out-of-sample.

Every score is reported two ways. **Grouped by run** keeps other runs of the
same config in training, so it measures recognising a known workload.
**Grouped by workload** holds out entire configs, so it measures
generalisation. Quote the grouped-by-workload number.

## Live dashboard

Classifies a rolling 30 s window once per second and shows how long detection
takes, what happens across a workload transition, and how false alarms pile
up. Two models can score the same window side by side. See
[docs/INTERPRETING.md](docs/INTERPRETING.md) before trusting any panel.

```bash
# offline replay, no GPU needed. Hold out a config, then replay it: a
# transition from inference to training the model has never seen.
python train_classifier.py workload --data data/workload --out models/holdout     --holdout-config train_resnet18_bs64 --holdout-config infer_resnet18_bs64
python train_classifier.py workload --data data/workload --out models/full
python -m dashboard --source replay --model models/holdout/threeway.joblib     --compare-model models/full/threeway.joblib     --trace data/workload/infer_resnet18_bs64_<...>.parquet     --trace data/workload/train_resnet18_bs64_<...>.parquet     --speed 10

# live, on the GPU machine; ground truth read from a running collection
python -m dashboard --source nvml --gpu 0 --model models/full/threeway.joblib --track-collection
```

Open <http://127.0.0.1:8000>. Add `--model-id-model models/<dir>/model_id_no_memory.joblib`
for the "which LLM is running" panel. The server has no authentication and
binds to loopback by default. The banner turns red if the model's power regime
or GPU doesn't match the source, or if a replayed trace was in the model's
training data.

## Rules that silently corrupt data if broken

1. **Labels decide classes, by substring.** A label containing `train` is
   `ml_training`. Otherwise one containing `infer` is `ml_inference`. Anything
   else is `other`. Check new labels with `run_workloads.py --list`.
2. **Never rename a trace.** Labels are parsed from filenames
   (`<label>_NVIDIA_...parquet`) when the column is missing.
3. **Don't pool power regimes.** The 200 W-capped and 350 W data differ by
   about 150 W in the most important signals, so a pooled model learns which
   machine a trace came from. Train on one regime directory at a time.
   `metadata.json` records peak power and GPU name so you can check.
4. **Idle windows have constant signals.** Skew and kurtosis are undefined, so
   they're zero-filled when the feature matrix is built.
5. **LLM workloads pin `min_new_tokens`.** Without it, a model that emits EOS
   early does less work, and its telemetry reflects EOS behaviour rather than
   the model.

## Results on the author's data (not published)

| corpus | task | grouped by run | grouped by workload |
|---|---|---|---|
| 200 W, 105 runs / 35 configs | binary | 0.986 | **0.831** |
| | three-way | 0.970 | **0.635** |
| 200 W, original 90-run subset | binary | 0.959 | **0.906** |
| | three-way | 0.987 | **0.730** |

The 105-run corpus includes the `gradprobe_*` configs (backward passes with no
weight updates) and `sci_*` scientific configs. These are designed to look
like training without being training, and they lower the held-out scores.

Model ID (10 LLMs, batch 4, grouped by run): 0.999 with memory features,
0.936 with every memory-derived feature removed, 0.986 from
`mem_used_mb_mean` alone. At a fixed serving config, memory footprint alone
nearly identifies the model. On the batch sweep, holding out a whole batch
size drops accuracy to 0.63 (0.56 without memory). Model identity mostly does
not carry over to a new serving configuration.

## Limitations

- **Small.** 3 runs per config. A perfect score means no errors across these
  windows, not a measured error rate.
- **One GPU model** (RTX 3090). No adversarial or evasive workloads.
- **Synthetic vision training** reuses one random batch, so there is no epoch
  structure.
- **Reimplemented features.** The original results used an external research
  repo's feature pipeline, which isn't included here. `gputel/features.py` is
  a reimplementation of its 159 window-local features, going by their names.
  Scores are close to the originals but not identical (90-run subset:
  0.906 / 0.730 here vs 0.932 / 0.805 originally).
- **Reimplemented non-ML workloads.** `fft`, `nbody`, `mining`, `render` and
  `idle` are new implementations. The `sci_*` configs in the 200 W data have
  no workload code, so they can't be reproduced from this repo.
