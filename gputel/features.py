"""Trace loading, labelling, windowing and feature extraction.

Every trace is a 1 Hz parquet written by collect_telemetry.py. Each is cut into
fixed windows (30 s / 15 s stride by default) and every window becomes one row
of 159 window-local features: 13 statistics over each of the nine NVML signals,
plus autocorrelation, spectral, duty-cycle, memory-trend, PCIe and codec
features.

Only window-local features are computed. Anything that needs a whole run (e.g.
"memory growth in the first 30 s of the run") cannot be produced for a rolling
window and was measured to add nothing on this corpus, so it is left out.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

# The nine signals the classifier reads.
SIGNALS = [
    "gpu_utilization_pct", "mem_utilization_pct", "mem_used_mb",
    "power_draw_w", "temperature_c", "sm_clock_mhz", "mem_clock_mhz",
    "pcie_tx_mbps", "pcie_rx_mbps",
]
CODEC = ["encoder_util_pct", "decoder_util_pct"]
META_COLS = ["run_id", "workload_label", "threeway_label", "window_start"]

# Features derived from resident memory. Memory footprint is close to a direct
# readout of parameter count, so model-ID is also scored with these removed.
MEMORY_PREFIXES = ("mem_used_mb", "mem_utilization", "acf1_mem", "acf2_mem",
                   "acf5_mem", "acf10_mem", "acf20_mem", "mem_slope",
                   "mem_second_half", "mem_used_fft")

ACF_LAGS = (1, 2, 5, 10, 20)
PCIE_ACTIVE_MBPS = 1.0   # an idle 3090 still shows ~0.3-0.7 MB/s of PCIe chatter
MIN_WINDOW_SAMPLES = 5


# ── labels ───────────────────────────────────────────────────────────────────

def threeway_label(label: str) -> str:
    """ml_training | ml_inference | other, from a workload label.

    Labels are matched by substring, "train" first. Pick new labels so this
    returns the class you intend: e.g. `llm_infer_*` is inference, and anything
    without "train" or "infer" in it is `other`.
    """
    lc = label.lower()
    if "train" in lc:
        return "ml_training"
    if "infer" in lc:
        return "ml_inference"
    return "other"


def batch_size(label: str) -> int | None:
    """Serving batch size when the label carries a `_bs<N>` token."""
    for tok in reversed(label.split("_")):
        if tok.startswith("bs") and tok[2:].isdigit():
            return int(tok[2:])
    return None


def model_label(label: str) -> str:
    """Model name from an LLM label: llm_infer_opt_1p3b -> opt_1p3b,
    llm_infer_sweep_opt_bs4 -> opt."""
    for prefix in ("llm_infer_sweep_", "llm_infer_"):
        if label.startswith(prefix):
            label = label[len(prefix):]
            break
    parts = label.split("_")
    if batch_size(label) is not None and parts[-1].startswith("bs"):
        parts = parts[:-1]
    return "_".join(parts)


def label_from_path(path: Path) -> str:
    """Filenames are `<label>_<GPU name>_<run>_<timestamp>.parquet`.

    Never rename a trace: when the label column is absent this is where the
    label comes from, and a tidied filename becomes a mislabelled sample.
    """
    stem = path.stem
    return stem.split("_NVIDIA_")[0] if "_NVIDIA_" in stem else stem


# ── loading ──────────────────────────────────────────────────────────────────

def load_trace(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_parquet(path)
    if "workload_label" not in df.columns:
        df["workload_label"] = label_from_path(path)
    if "run_id" not in df.columns:
        df["run_id"] = path.stem
    for col in SIGNALS + CODEC:
        if col not in df.columns:
            df[col] = 0.0
    df["threeway_label"] = df["workload_label"].map(threeway_label)
    return df.sort_values("timestamp_epoch").reset_index(drop=True)


def load_traces(data_dir: str | Path, min_samples: int = 0) -> pd.DataFrame:
    """Every .parquet under data_dir (recursive), one run per file."""
    frames, short = [], []
    for path in sorted(Path(data_dir).rglob("*.parquet")):
        if path.read_bytes()[:4] != b"PAR1":
            raise ValueError(f"{path} is not a parquet file (a Git LFS pointer?)")
        df = load_trace(path)
        if len(df) < min_samples:
            short.append((path.name, len(df)))
            continue
        frames.append(df)
    for name, n in short:
        print(f"  skipping truncated run ({n} samples): {name}")
    if not frames:
        raise SystemExit(f"no usable .parquet traces under {data_dir}")
    return pd.concat(frames, ignore_index=True)


# ── features ─────────────────────────────────────────────────────────────────

def _stats(prefix: str, x: np.ndarray) -> dict:
    mean, std = x.mean(), x.std()
    p25, p50, p75, p95 = np.percentile(x, [25, 50, 75, 95])
    with warnings.catch_warnings():
        # Constant signals (an idle GPU) have undefined skew/kurtosis; NaN is
        # zero-filled when the feature matrix is built.
        warnings.simplefilter("ignore", RuntimeWarning)
        sk = skew(x) if std > 0 else np.nan
        ku = kurtosis(x) if std > 0 else np.nan
    return {
        f"{prefix}_mean": mean, f"{prefix}_std": std,
        f"{prefix}_min": x.min(), f"{prefix}_max": x.max(),
        f"{prefix}_p25": p25, f"{prefix}_p50": p50,
        f"{prefix}_p75": p75, f"{prefix}_p95": p95,
        f"{prefix}_iqr": p75 - p25, f"{prefix}_range": x.max() - x.min(),
        f"{prefix}_cv": std / mean if mean else 0.0,
        f"{prefix}_skew": sk, f"{prefix}_kurt": ku,
    }


def _acf(x: np.ndarray, lag: int) -> float:
    if len(x) <= lag + 1:
        return np.nan
    a, b = x[:-lag] - x.mean(), x[lag:] - x.mean()
    denom = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / denom) if denom > 0 else np.nan


def _fft_peak(x: np.ndarray) -> tuple[float, float]:
    """(period in samples, share of spectral power in the peak bin), DC excluded."""
    if len(x) < 4 or x.std() == 0:
        return 0.0, 0.0
    spec = np.abs(np.fft.rfft(x - x.mean())) ** 2
    freqs = np.fft.rfftfreq(len(x))
    spec, freqs = spec[1:], freqs[1:]
    i = int(spec.argmax())
    return float(1.0 / freqs[i]), float(spec[i] / spec.sum())


def _slope(y: np.ndarray) -> float:
    if len(y) < 2:
        return 0.0
    return float(np.polyfit(np.arange(len(y)), y, 1)[0])


def extract_features(w: pd.DataFrame) -> dict:
    """The 159 window-local features for one window of 1 Hz samples."""
    s = {c: w[c].to_numpy(dtype=float) for c in SIGNALS + CODEC}
    util, power, mem = s["gpu_utilization_pct"], s["power_draw_w"], s["mem_used_mb"]
    pcie = s["pcie_tx_mbps"] + s["pcie_rx_mbps"]
    n = len(w)

    f: dict = {}
    for c in SIGNALS:
        f.update(_stats(c, s[c]))

    f["power_per_util"] = power.mean() / max(util.mean(), 1.0)
    f["pcie_total_mean"] = pcie.mean()
    f["util_per_sm_pct"] = util.mean() / max(s["sm_clock_mhz"].mean() / 1000.0, 1e-3)

    for lag in ACF_LAGS:
        f[f"acf{lag}_gpu_util"] = _acf(util, lag)
        f[f"acf{lag}_power"] = _acf(power, lag)
        f[f"acf{lag}_mem_used"] = _acf(mem, lag)

    half = n // 2
    first, second = _slope(mem[:half]), _slope(mem[half:])
    f["mem_slope"] = _slope(mem)
    f["mem_slope_first_half"] = first
    f["mem_slope_second_half"] = second
    f["mem_slope_ratio"] = second / first if first else 0.0
    m2 = mem[half:]
    f["mem_second_half_cv"] = m2.std() / m2.mean() if m2.mean() else 0.0

    f["gpu_util_fft_period"], f["gpu_util_fft_peak_power"] = _fft_peak(util)
    f["power_fft_period"], f["power_fft_peak_power"] = _fft_peak(power)
    f["mem_used_fft_peak_power"] = _fft_peak(mem)[1]

    busy = util >= 50
    f["duty_cycle_80"] = float((util >= 80).mean())
    f["idle_frac"] = float((util < 5).mean())
    f["util_transitions_per_sec"] = float(np.count_nonzero(np.diff(busy))) / max(n - 1, 1)
    us, um = util.std(), util.mean()
    f["util_burstiness"] = (us - um) / (us + um) if (us + um) else 0.0

    f["power_cv"] = power.std() / power.mean() if power.mean() else 0.0
    f["power_rolling_var"] = float(pd.Series(power).rolling(5, min_periods=2).var().mean())
    f["power_range_frac"] = (power.max() - power.min()) / power.max() if power.max() else 0.0

    f["pcie_cv"] = pcie.std() / pcie.mean() if pcie.mean() else 0.0
    f["pcie_fft_peak_power"] = _fft_peak(pcie)[1]
    f["pcie_nonzero_frac"] = float((pcie > PCIE_ACTIVE_MBPS).mean())

    for c in CODEC:
        f[f"{c}_mean"] = s[c].mean()
        f[f"{c}_nonzero_frac"] = float((s[c] > 0).mean())
    return f


def sliding_windows(raw: pd.DataFrame, window_sec: float = 30,
                    stride_sec: float = 15) -> pd.DataFrame:
    """One feature row per (run, window). Windows never span two runs."""
    rows = []
    for run_id, run in raw.groupby("run_id", sort=False):
        t = run["timestamp_epoch"].to_numpy()
        start = t[0]
        while start + window_sec <= t[-1] + 1:
            mask = (t >= start) & (t < start + window_sec)
            if mask.sum() >= MIN_WINDOW_SAMPLES:
                w = run[mask]
                feats = extract_features(w)
                feats.update(run_id=run_id,
                             workload_label=w["workload_label"].iloc[0],
                             threeway_label=w["threeway_label"].iloc[0],
                             window_start=start)
                rows.append(feats)
            start += stride_sec
    return pd.DataFrame(rows)


def feature_cols(windows: pd.DataFrame) -> list[str]:
    return [c for c in windows.columns if c not in META_COLS]


def to_matrix(windows: pd.DataFrame, cols: list[str]) -> np.ndarray:
    X = windows.reindex(columns=cols).to_numpy(dtype=np.float32)
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
