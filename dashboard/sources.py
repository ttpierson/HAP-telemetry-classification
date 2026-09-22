"""Telemetry sources: live NVML, or replay of a saved trace.

Both yield the trace row schema written by collect_telemetry.py, so the ring
buffer, feature extraction and classifier see identical input whether samples
come from a GPU or from a parquet file on a laptop. That is what makes replay
a real test of the live path rather than a mock.
"""
from __future__ import annotations

import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from gputel.features import CODEC, SIGNALS, label_from_path

RAW_SIGNALS = SIGNALS


@dataclass
class SourceInfo:
    """Static description of where samples are coming from."""
    kind: str                       # "nvml" | "replay"
    host: str
    gpu_name: str | None = None
    gpu_uuid: str | None = None
    gpu_index: int | None = None
    power_limit_w: float | None = None
    driver_version: str | None = None
    detail: str = ""
    trace_names: list[str] = field(default_factory=list)
    # Peak power seen in the data. For replay this is the only clue to the
    # power regime the trace was collected under.
    observed_max_power_w: float | None = None
    # run_ids of replayed traces, so the model can be asked if it trained on them.
    run_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return dict(kind=self.kind, host=self.host, gpu_name=self.gpu_name,
                    gpu_uuid=self.gpu_uuid, gpu_index=self.gpu_index,
                    power_limit_w=self.power_limit_w,
                    driver_version=self.driver_version, detail=self.detail,
                    trace_names=self.trace_names,
                    observed_max_power_w=self.observed_max_power_w,
                    run_ids=self.run_ids)


class TelemetrySource:
    """A 1 Hz source of telemetry rows."""

    info: SourceInfo
    interval = 1.0

    # Whether this source knows what the GPU is really doing. A replayed trace
    # carries its own label; a live GPU does not, so the engine must not let
    # the source overwrite an operator's mark.
    provides_truth = False

    def sample(self) -> dict | None:
        """One reading. None means the source is exhausted (replay end)."""
        raise NotImplementedError

    def true_label(self) -> str | None:
        return None

    def close(self) -> None:
        pass


class NvmlSource(TelemetrySource):
    """Polls one explicitly chosen GPU via NVML."""

    def __init__(self, gpu_index: int):
        from gputel.nvml import Gpu
        self.gpu = Gpu(gpu_index)
        limit = self.gpu.power_limit_w
        if limit is None:
            raise RuntimeError(
                f"GPU {gpu_index} ({self.gpu.name}) reports no enforced power limit, "
                "which usually means the device is unhealthy. Pick another GPU.")
        self.info = SourceInfo(
            kind="nvml", host=socket.gethostname(),
            gpu_name=self.gpu.name, gpu_uuid=self.gpu.uuid, gpu_index=gpu_index,
            power_limit_w=limit, driver_version=self.gpu.driver,
            detail=f"NVML, GPU {gpu_index}, {limit:.0f} W cap")
        self._peak = 0.0
        self._n = 0

    def sample(self) -> dict:
        row = self.gpu.sample()
        self._peak = max(self._peak, row["power_draw_w"])
        self.info.observed_max_power_w = round(self._peak, 1)
        # Re-read the power cap periodically: it can change under a running
        # process, and a provenance check that only looks once would miss it.
        self._n += 1
        if self._n % 30 == 0:
            self.info.power_limit_w = self.gpu.power_limit_w or self.info.power_limit_w
        return row

    def close(self) -> None:
        self.gpu.close()


class ReplaySource(TelemetrySource):
    """Replays saved .parquet traces through the same path as live telemetry.

    Chaining several traces gives the transition panel a workload change with
    known ground truth. Timestamps are rewritten to one contiguous 1 Hz stream
    that advances 1 s per sample regardless of replay speed, so every latency
    is reported in telemetry seconds.
    """

    provides_truth = True

    def __init__(self, paths: list[str | Path], speed: float = 1.0, loop: bool = False):
        self.speed = max(speed, 0.01)
        self.loop = loop
        self._rows: list[dict] = []
        self._labels: list[str] = []
        names, run_ids = [], []

        for path in map(Path, paths):
            df = pd.read_parquet(path)
            label = (str(df["workload_label"].iloc[0]) if "workload_label" in df.columns
                     else label_from_path(path))
            names.append(f"{label} ({len(df)}s)")
            run_ids.append(str(df["run_id"].iloc[0]) if "run_id" in df.columns else path.stem)
            for col in SIGNALS + CODEC + ["mem_total_mb"]:
                if col not in df.columns:
                    df[col] = 0.0
            for rec in df.to_dict("records"):
                self._rows.append(rec)
                self._labels.append(label)
        if not self._rows:
            raise ValueError("no rows loaded from the given traces")

        self._i = 0
        self._current_label: str | None = None
        self._t0 = datetime.now(timezone.utc).timestamp()
        self._emitted = 0.0

        gpu_name = self._rows[0].get("gpu_name")
        # Peak draw under load only: an idle trace never nears the cap, so it
        # says nothing about the regime it came from.
        loaded = [float(r.get("power_draw_w") or 0.0) for r in self._rows
                  if float(r.get("gpu_utilization_pct") or 0.0) >= 50]
        peak = max(loaded) if loaded else None
        self.info = SourceInfo(
            kind="replay", host=socket.gethostname(),
            gpu_name=str(gpu_name) if gpu_name else None,
            detail=f"replay of {len(names)} trace(s) at {self.speed:g}x",
            trace_names=names, run_ids=run_ids,
            observed_max_power_w=round(peak, 1) if peak else None)

    @property
    def interval(self) -> float:
        return 1.0 / self.speed

    def sample(self) -> dict | None:
        if self._i >= len(self._rows):
            if not self.loop:
                return None
            self._i = 0
        rec = dict(self._rows[self._i])
        self._current_label = self._labels[self._i]
        self._i += 1
        stamp = self._t0 + self._emitted
        self._emitted += 1.0
        rec["timestamp_utc"] = datetime.fromtimestamp(stamp, timezone.utc).isoformat()
        rec["timestamp_epoch"] = stamp
        for col in SIGNALS + CODEC + ["mem_total_mb"]:
            rec[col] = float(rec.get(col) or 0.0)
        return rec

    def true_label(self) -> str | None:
        return self._current_label
