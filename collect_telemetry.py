#!/usr/bin/env python3
"""Record 1 Hz NVML telemetry from one GPU into a labelled parquet trace.

Either record for a fixed time:

    python collect_telemetry.py --gpu 0 --label idle --duration 600

or wrap a command, which is pinned to the same GPU and recorded until it exits:

    python collect_telemetry.py --gpu 0 --label train_resnet18_bs64 -- \\
        python -m gputel.workloads vision --mode train --model resnet18

Output: <output-dir>/<label>_<GPU name>_<run id>_<UTC timestamp>.parquet, one
row per sample. The label is parsed back out of the filename when training, so
never rename a trace.

Telemetry is whole-GPU: anything else running on the device is recorded under
your label with no indication. The script warns if the GPU is already busy.
"""
from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gputel.nvml import Gpu  # noqa: E402


def record(gpu: Gpu, hz: float, until) -> list[dict]:
    """Sample on absolute deadlines until until() is true.

    Deadlines are start + n*period rather than last + period, so per-sample
    overhead never accumulates into drift; a late sample skips missed ticks
    instead of firing a catch-up burst.
    """
    period, rows = 1.0 / hz, []
    start, n = time.monotonic(), 0
    while not until():
        rows.append(gpu.sample())
        n += 1
        next_t = start + n * period
        now = time.monotonic()
        if now > next_t:
            n += int((now - next_t) // period) + 1
            next_t = start + n * period
        time.sleep(max(0.0, next_t - time.monotonic()))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", type=int, required=True, help="NVML index to record")
    ap.add_argument("--label", required=True,
                    help="workload label; decides the training class (see README)")
    ap.add_argument("--output-dir", default="data/new")
    ap.add_argument("--duration", type=float, default=None,
                    help="seconds to record (required without a command; with a "
                         "command, an upper bound)")
    ap.add_argument("--hz", type=float, default=1.0)
    ap.add_argument("--baseline", type=float, default=2.0,
                    help="seconds recorded before the command starts")
    ap.add_argument("--log", default=None, help="file for the command's output")
    ap.add_argument("command", nargs=argparse.REMAINDER,
                    help="-- followed by a command to run on the GPU")
    a = ap.parse_args(argv)

    cmd = a.command[1:] if a.command[:1] == ["--"] else a.command
    if not cmd and a.duration is None:
        ap.error("give --duration, or a command after --")
    if not a.label.strip() or "_NVIDIA_" in a.label:
        ap.error("label must be non-empty and must not contain '_NVIDIA_'")

    gpu = Gpu(a.gpu)
    others = gpu.busy_pids()
    if others:
        print(f"WARNING: GPU {a.gpu} already runs compute process(es) {others}; "
              "their activity will be recorded under this label.", file=sys.stderr)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    t0 = time.monotonic()
    deadline = t0 + a.duration if a.duration else float("inf")
    proc, rc = None, 0
    try:
        if cmd:
            rows = record(gpu, a.hz, lambda: stop["flag"] or time.monotonic() - t0 >= a.baseline)
            # PCI_BUS_ID ordering makes CUDA device numbers match NVML's.
            env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID",
                       CUDA_VISIBLE_DEVICES=str(a.gpu))
            out = open(a.log, "w") if a.log else None
            proc = subprocess.Popen(cmd, env=env, stdout=out, stderr=subprocess.STDOUT if out else None)
            rows += record(gpu, a.hz, lambda: stop["flag"] or proc.poll() is not None
                           or time.monotonic() >= deadline)
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(30)
                except subprocess.TimeoutExpired:
                    proc.kill()
            rc = proc.wait()
            if out:
                out.close()
        else:
            rows = record(gpu, a.hz, lambda: stop["flag"] or time.monotonic() >= deadline)
    finally:
        gpu.close()

    if not rows:
        print("no samples recorded", file=sys.stderr)
        return 1
    run_id = uuid.uuid4().hex[:12]
    df = pd.DataFrame(rows)
    df["workload_label"] = a.label
    df["run_id"] = run_id
    df["gpu_name"] = gpu.name
    df["gpu_uuid"] = gpu.uuid
    df["driver_version"] = gpu.driver
    df["power_limit_w"] = gpu.power_limit_w

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(a.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{a.label}_{gpu.name.replace(' ', '_')}_{run_id[:8]}_{stamp}.parquet"
    df.to_parquet(path, index=False)
    print(f"{len(df)} samples -> {path}" + (f"  (command exit {rc})" if cmd else ""))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
