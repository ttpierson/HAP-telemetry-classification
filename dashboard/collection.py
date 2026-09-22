"""Read ground truth off the collection job's process table.

run_workloads.py records each run as

    collect_telemetry.py --gpu <n> --label <label> ... -- <the actual workload>

so while a collection is in progress the host already knows what every GPU is
doing, and it knows it in exactly the vocabulary the classifier was trained on.
Scraping that beats asking an operator to keep a dropdown in sync with a job
that changes workload every ten minutes.

When no run targets our GPU, the GPU is genuinely idle, and saying so is itself
ground truth: it is what lets the false-positive ledger count the gaps between
runs.

This reads /proc directly rather than shelling out to ps once a second, and it
only ever reads. Nothing here signals, kills, or otherwise touches another
process.
"""
from __future__ import annotations

import logging
from pathlib import Path

log = logging.getLogger("dashboard.collection")

MARKER = "collect_telemetry.py"


class CollectionTracker:
    """Reports the workload label the collection job is running on one GPU."""

    def __init__(self, gpu_index: int, idle_label: str = "idle"):
        self.gpu_index = gpu_index
        self.idle_label = idle_label
        self._last: str | None = None

    def current_label(self) -> str:
        """The label for our GPU right now, or the idle label if nothing runs."""
        for cmdline in _iter_cmdlines():
            joined = " ".join(cmdline)
            if MARKER not in joined:
                continue
            label = _arg(cmdline, "--label")
            idx = _arg(cmdline, "--gpu")
            if label is None or idx is None:
                continue
            try:
                if int(idx) != self.gpu_index:
                    continue
            except ValueError:
                continue
            if label != self._last:
                log.info("collection now running %r on gpu %d", label, self.gpu_index)
                self._last = label
            return label

        if self._last != self.idle_label:
            log.info("no collection run on gpu %d — idle", self.gpu_index)
            self._last = self.idle_label
        return self.idle_label


def _iter_cmdlines():
    if not Path("/proc").is_dir():
        return  # Linux only
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            raw = (entry / "cmdline").read_bytes()
        except (OSError, PermissionError):
            continue  # not ours to read, or gone between listing and reading
        if not raw:
            continue
        yield [p.decode("utf-8", "replace") for p in raw.split(b"\0") if p]


def _arg(cmdline: list[str], flag: str) -> str | None:
    """Value of `--flag value` or `--flag=value`, whichever form was used."""
    if "--" in cmdline:
        cmdline = cmdline[:cmdline.index("--")]
    for i, tok in enumerate(cmdline):
        if tok == flag and i + 1 < len(cmdline):
            return cmdline[i + 1]
        if tok.startswith(flag + "="):
            return tok.split("=", 1)[1]
    return None
