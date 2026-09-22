#!/usr/bin/env python3
"""Run the workload matrix across GPUs, recording telemetry for every run.

One worker per GPU pulls (config, repetition) tasks from a shared queue and
runs each through collect_telemetry.py, so every run produces one labelled
trace. Progress is checkpointed after each run: re-issuing the same command
resumes where it stopped.

    python run_workloads.py --list
    python run_workloads.py --gpus 0,1 --dry-run
    python run_workloads.py --gpus 0,1 --duration 600 --reps 3
    python run_workloads.py --gpus 0 --only llm_infer_sweep --output-dir data/sweep

GPUs are never auto-enumerated: pass the ones that are yours to use.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gputel.features import threeway_label  # noqa: E402

HERE = Path(__file__).resolve().parent
PY = sys.executable
W = [PY, "-m", "gputel.workloads"]


def build_matrix() -> list[tuple[str, list[str]]]:
    """(label, workload command). The label decides the training class."""
    m = []

    def vision(label, mode, model, bs, amp=False):
        m.append((label, W + ["vision", "--mode", mode, "--model", model,
                              "--batch-size", str(bs)] + (["--amp"] if amp else [])))

    # ml_training: architecture x batch size x precision
    vision("train_resnet18_bs32", "train", "resnet18", 32)
    vision("train_resnet18_bs64", "train", "resnet18", 64)
    vision("train_resnet18_bs128", "train", "resnet18", 128)
    vision("train_resnet18_amp_bs64", "train", "resnet18", 64, amp=True)
    vision("train_resnet50_bs32", "train", "resnet50", 32)
    vision("train_resnet50_bs64", "train", "resnet50", 64)
    vision("train_resnet50_amp_bs64", "train", "resnet50", 64, amp=True)
    vision("train_mobilenetv3_bs64", "train", "mobilenet_v3_large", 64)
    vision("train_mobilenetv3_bs128", "train", "mobilenet_v3_large", 128)
    vision("train_vitb16_bs32", "train", "vit_b_16", 32)
    vision("train_vitb16_amp_bs32", "train", "vit_b_16", 32, amp=True)

    # ml_inference: the same architectures, forward only
    vision("infer_resnet18_bs32", "infer", "resnet18", 32)
    vision("infer_resnet18_bs64", "infer", "resnet18", 64)
    vision("infer_resnet18_bs256", "infer", "resnet18", 256)
    vision("infer_resnet18_amp_bs64", "infer", "resnet18", 64, amp=True)
    vision("infer_resnet50_bs64", "infer", "resnet50", 64)
    vision("infer_resnet50_bs128", "infer", "resnet50", 128)
    vision("infer_mobilenetv3_bs128", "infer", "mobilenet_v3_large", 128)
    vision("infer_vitb16_bs64", "infer", "vit_b_16", 64)

    # other: varied non-ML GPU work
    m += [
        ("idle", W + ["idle"]),
        ("cufft_4096", W + ["fft", "--size", "4096"]),
        ("cufft_2048", W + ["fft", "--size", "2048"]),
        ("nbody_16384", W + ["nbody", "--particles", "16384"]),
        ("nbody_8192", W + ["nbody", "--particles", "8192"]),
        ("mining_ethash_proxy", W + ["mining"]),
        ("rendering_proxy", W + ["render"]),
    ]

    # other: backward passes that are not training. Tests whether a classifier
    # keys on "a backward pass is happening" rather than "a model is learning".
    for label, mode in (("gradprobe_pgd", "pgd"),
                        ("gradprobe_integrated_gradients", "ig"),
                        ("gradprobe_feature_viz", "featviz"),
                        ("gradprobe_pinn_residual", "pinn")):
        m.append((label, W + ["gradprobe", "--mode", mode]))

    # ml_inference, model identification: ten open LLMs, eight architectures,
    # mostly in a 1-2B band so architecture rather than size is the variable.
    llms = [
        ("llm_infer_tinyllama_1p1b", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),   # llama
        ("llm_infer_smollm2_1p7b", "HuggingFaceTB/SmolLM2-1.7B-Instruct"),    # llama
        ("llm_infer_qwen25_1p5b", "Qwen/Qwen2.5-1.5B-Instruct"),              # qwen2
        ("llm_infer_pythia_1p4b", "EleutherAI/pythia-1.4b"),                  # gpt_neox
        ("llm_infer_falcon_1b", "tiiuae/falcon-rw-1b"),                       # falcon
        ("llm_infer_stablelm2_1p6b", "stabilityai/stablelm-2-1_6b"),          # stablelm
        ("llm_infer_granite_2b", "ibm-granite/granite-3.1-2b-instruct"),      # granite
        ("llm_infer_opt_1p3b", "facebook/opt-1.3b"),                          # opt
        ("llm_infer_phi3_mini", "microsoft/Phi-3-mini-4k-instruct"),          # phi3
        ("llm_infer_mistral_7b", "mistralai/Mistral-7B-Instruct-v0.3"),       # mistral
    ]
    for label, model in llms:
        m.append((label, W + ["llm", "--model", model, "--batch-size", "4"]))

    # ml_inference, model x batch-size factorial: four models within 1.31-1.54B
    # crossed with four batch sizes, so memory footprint overlaps across cells.
    sweep = [("falcon", "tiiuae/falcon-rw-1b"), ("opt", "facebook/opt-1.3b"),
             ("pythia", "EleutherAI/pythia-1.4b"), ("qwen25", "Qwen/Qwen2.5-1.5B-Instruct")]
    for short, model in sweep:
        for bs in (1, 4, 16, 32):
            m.append((f"llm_infer_sweep_{short}_bs{bs}",
                      W + ["llm", "--model", model, "--batch-size", str(bs)]))
    return m


# ── GPU health ───────────────────────────────────────────────────────────────

def gpu_health(idx: int) -> str:
    """"ok" | "busy" (someone else's process) | "dead".

    NVML and CUDA are separate: a GPU can enumerate under NVML and still be
    unusable by CUDA, so a real kernel is run in an isolated subprocess too.
    """
    try:
        import pynvml
        pynvml.nvmlInit()
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            if pynvml.nvmlDeviceGetComputeRunningProcesses(h):
                return "busy"
        finally:
            pynvml.nvmlShutdown()
    except Exception:
        return "dead"
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(idx))
    try:
        r = subprocess.run([PY, "-c", "import torch;a=torch.randn(64,64,device='cuda');"
                                      "(a@a).sum().item()"],
                           capture_output=True, text=True, timeout=180, env=env)
    except subprocess.TimeoutExpired:
        return "dead"
    return "ok" if r.returncode == 0 else "dead"


# ── dispatch ─────────────────────────────────────────────────────────────────

class Progress:
    def __init__(self, path: Path, fresh: bool):
        self.path, self.lock = path, threading.Lock()
        self.data = {"done": [], "failed": []}
        if path.exists() and not fresh:
            self.data = json.loads(path.read_text())

    def record(self, key: str, ok: bool, **info):
        with self.lock:
            if ok:
                self.data["done"].append(key)
            else:
                self.data["failed"].append({"key": key, **info})
            self.path.write_text(json.dumps(self.data, indent=2))


def worker(gpu, q, prog, a, logdir):
    """Runs tasks on one GPU. Retires if the GPU dies or stays busy; the task
    in hand goes back on the queue for a healthy worker."""
    busy_waits = failures = 0
    while True:
        try:
            label, cmd, rep = q.get_nowait()
        except queue.Empty:
            return
        key = f"{label}|rep{rep}"

        health = gpu_health(gpu)
        if health != "ok":
            q.put((label, cmd, rep))
            if health == "dead":
                print(f"[gpu{gpu}] RETIRING: GPU not usable", flush=True)
                return
            busy_waits += 1
            if busy_waits > a.max_busy_waits:
                print(f"[gpu{gpu}] RETIRING: busy for {busy_waits} checks", flush=True)
                return
            print(f"[gpu{gpu}] busy with another process, waiting", flush=True)
            time.sleep(a.busy_wait_sec)
            continue
        busy_waits = 0

        log = logdir / f"{label}_rep{rep}_gpu{gpu}.log"
        launch = [PY, str(HERE / "collect_telemetry.py"), "--gpu", str(gpu),
                  "--label", label, "--output-dir", a.output_dir,
                  "--duration", str(a.duration + a.startup_allowance),
                  "--log", str(log), "--"] + cmd + ["--duration", str(a.duration)]
        print(f"[gpu{gpu}] START {key}", flush=True)
        t0 = time.time()
        rc = subprocess.run(launch, cwd=HERE).returncode
        if rc == 0:
            failures = 0
            prog.record(key, True)
            print(f"[gpu{gpu}] OK    {key} ({time.time() - t0:.0f}s)", flush=True)
        else:
            failures += 1
            prog.record(key, False, gpu=gpu, rc=rc, log=str(log))
            print(f"[gpu{gpu}] FAIL  {key} rc={rc}, see {log}", flush=True)
            if failures >= a.max_failures:
                print(f"[gpu{gpu}] RETIRING: {failures} consecutive failures", flush=True)
                return
        time.sleep(a.gap)   # let the GPU settle before the next run's baseline


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", help="comma-separated NVML indices, e.g. 0,1")
    ap.add_argument("--duration", type=int, default=600, help="seconds per run")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--only", default=None, help="only configs whose label contains this")
    ap.add_argument("--output-dir", default="data/new")
    ap.add_argument("--gap", type=int, default=10, help="idle seconds between runs")
    ap.add_argument("--startup-allowance", type=int, default=300,
                    help="extra seconds allowed for model loading before a run is killed")
    ap.add_argument("--busy-wait-sec", type=int, default=60)
    ap.add_argument("--max-busy-waits", type=int, default=30)
    ap.add_argument("--max-failures", type=int, default=3)
    ap.add_argument("--fresh", action="store_true", help="ignore saved progress")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--list", action="store_true", help="print the matrix and exit")
    a = ap.parse_args(argv)

    matrix = build_matrix()
    if a.only:
        matrix = [(l, c) for l, c in matrix if a.only in l]
        if not matrix:
            ap.error(f"--only {a.only!r} matched no configs")
    if a.list:
        for label, cmd in matrix:
            print(f"{label:34s} {threeway_label(label):13s} {' '.join(cmd[2:])}")
        return 0
    if not a.gpus:
        ap.error("--gpus is required")
    gpus = [int(g) for g in a.gpus.split(",") if g.strip()]

    out = Path(a.output_dir).resolve()
    a.output_dir = str(out)   # the collector runs with cwd=HERE
    out.mkdir(parents=True, exist_ok=True)
    logdir = out / "logs"
    logdir.mkdir(exist_ok=True)
    prog = Progress(out / "progress.json", a.fresh)
    done = set(prog.data["done"])
    tasks = [(l, c, r) for r in range(1, a.reps + 1) for l, c in matrix
             if f"{l}|rep{r}" not in done]

    hours = len(tasks) * (a.duration + a.gap) / 3600 / len(gpus)
    print(f"{len(matrix)} configs x {a.reps} reps; {len(done)} done, "
          f"{len(tasks)} queued on GPUs {gpus} (~{hours:.1f} h)")
    if a.dry_run:
        for l, c, r in tasks[:10]:
            print(f"  rep{r} {l}: {' '.join(c[2:])}")
        print(f"  ... and {max(0, len(tasks) - 10)} more")
        return 0

    q = queue.Queue()
    for t in tasks:
        q.put(t)
    threads = [threading.Thread(target=worker, args=(g, q, prog, a, logdir)) for g in gpus]
    t0 = datetime.now(timezone.utc)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print(f"\nfinished in {(datetime.now(timezone.utc) - t0).total_seconds() / 3600:.2f} h: "
          f"{len(prog.data['done'])} ok, {len(prog.data['failed'])} failed")
    if not q.empty():
        print(f"{q.qsize()} task(s) never ran (no healthy GPU left); re-run to resume.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
