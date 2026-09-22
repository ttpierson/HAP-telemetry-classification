"""One NVML device, read into the trace row schema.

Shared by collect_telemetry.py and the live dashboard, so a recorded trace and
a live stream contain identical columns in identical units.
"""
from __future__ import annotations

from datetime import datetime, timezone


class Gpu:
    """One NVML device, chosen explicitly by index. Never enumerates all GPUs."""

    def __init__(self, index: int):
        import pynvml
        self.nv = pynvml
        pynvml.nvmlInit()
        self.index = index
        self.h = pynvml.nvmlDeviceGetHandleByIndex(index)
        txt = lambda v: v.decode() if isinstance(v, bytes) else v  # noqa: E731
        self.name = txt(pynvml.nvmlDeviceGetName(self.h))
        self.uuid = txt(pynvml.nvmlDeviceGetUUID(self.h))
        self.driver = txt(pynvml.nvmlSystemGetDriverVersion())
        self.mem_total_mb = pynvml.nvmlDeviceGetMemoryInfo(self.h).total // 2 ** 20

    @property
    def power_limit_w(self) -> float | None:
        """Read fresh each time: a cap can be changed under a running process."""
        try:
            return self.nv.nvmlDeviceGetEnforcedPowerLimit(self.h) / 1000
        except self.nv.NVMLError:
            return None

    def busy_pids(self) -> list[int]:
        return [p.pid for p in self.nv.nvmlDeviceGetComputeRunningProcesses(self.h)]

    def _opt(self, fn, *args, default=0.0):
        try:
            return fn(self.h, *args)
        except self.nv.NVMLError:
            return default

    def sample(self) -> dict:
        nv, h = self.nv, self.h
        now = datetime.now(timezone.utc)
        util = nv.nvmlDeviceGetUtilizationRates(h)
        return {
            "timestamp_utc": now.isoformat(),
            "timestamp_epoch": now.timestamp(),
            "gpu_utilization_pct": util.gpu,
            "mem_utilization_pct": util.memory,
            "mem_used_mb": nv.nvmlDeviceGetMemoryInfo(h).used // 2 ** 20,
            "mem_total_mb": self.mem_total_mb,
            "power_draw_w": nv.nvmlDeviceGetPowerUsage(h) / 1000,
            "temperature_c": nv.nvmlDeviceGetTemperature(h, nv.NVML_TEMPERATURE_GPU),
            "sm_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_SM),
            "mem_clock_mhz": nv.nvmlDeviceGetClockInfo(h, nv.NVML_CLOCK_MEM),
            # NVML reports PCIe throughput in KB/s; traces are in MB/s.
            "pcie_tx_mbps": self._opt(nv.nvmlDeviceGetPcieThroughput,
                                      nv.NVML_PCIE_UTIL_TX_BYTES) / 1024,
            "pcie_rx_mbps": self._opt(nv.nvmlDeviceGetPcieThroughput,
                                      nv.NVML_PCIE_UTIL_RX_BYTES) / 1024,
            "encoder_util_pct": self._opt(nv.nvmlDeviceGetEncoderUtilization,
                                          default=(0, 0))[0],
            "decoder_util_pct": self._opt(nv.nvmlDeviceGetDecoderUtilization,
                                          default=(0, 0))[0],
            "fan_speed_pct": self._opt(nv.nvmlDeviceGetFanSpeed, default=0),
        }

    def close(self):
        try:
            self.nv.nvmlShutdown()
        except Exception:
            pass
