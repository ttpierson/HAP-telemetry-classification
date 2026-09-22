"""The 1 Hz loop: sample, buffer, classify, and keep the state the UI reads.

One classification per second over the trailing 30 s window. Measured cost is
~4 ms of feature extraction plus ~30 ms of predict_proba, so about 3% of the
budget; nothing here is optimised for speed, only for being legible.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque

import numpy as np

import pandas as pd

from gputel.features import CODEC, MIN_WINDOW_SAMPLES, SIGNALS, extract_features

from .events import EventTracker
from .model import LoadedModel
from .sources import RAW_SIGNALS, TelemetrySource

log = logging.getLogger("dashboard.engine")

SPARKLINE_SECONDS = 120
HISTORY_SECONDS = 900


class Engine:
    """Owns the polling thread, the ring buffer, and the current state."""

    def __init__(self, source: TelemetrySource, model: LoadedModel,
                 confidence: float = 0.80, buffer_seconds: int = 3600,
                 compare_model: LoadedModel | None = None,
                 truth_tracker=None,
                 model_id_model: LoadedModel | None = None,
                 model_id_batch_size: int | None = 4):
        self.source = source
        self.model = model
        # A second model scoring the identical window. Its whole purpose is to
        # make the by-run / by-workload gap visible: point a model that trained
        # on this workload and one that never saw it at the same telemetry and
        # the difference between them is the difference between a demo and a
        # deployment.
        self.compare_model = compare_model
        # An external source of ground truth for a live GPU -- currently the
        # collection job's own process table. When present it owns the
        # declaration and the operator's dropdown steps aside, because the two
        # would otherwise overwrite each other every second.
        self.truth_tracker = truth_tracker

        # A third classifier answering a different question -- which LLM is
        # running -- over the same window. Closed-set over its ten models, so
        # the tracker beside it records when its answer can mean anything.
        self.model_id_model = model_id_model
        self.model_id = None
        if model_id_model is not None:
            from .modelid import ModelIdTracker
            self.model_id = ModelIdTracker(model_id_model.classes,
                                           trained_batch_size=model_id_batch_size)

        self.window_sec = model.window_sec

        self.samples: deque[dict] = deque(maxlen=buffer_seconds)
        self.predictions: deque[dict] = deque(maxlen=HISTORY_SECONDS)
        self.events = EventTracker(window_sec=self.window_sec,
                                   confidence=confidence)

        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

        self.started_at: float | None = None
        # Timestamp of the newest sample. All ages, latencies and episode
        # durations are measured on this clock so that replaying at 50x still
        # reports latencies in telemetry seconds.
        self.stream_now: float | None = None
        self.exhausted = False
        self.last_error: str | None = None
        self.timing_ms = {"extract": None, "predict": None}

        self._interval = getattr(source, "interval", 1.0)

    @property
    def provenance_warnings(self) -> list[dict]:
        """Checked against the GPU as it is now, not as it was at startup."""
        return self.model.compare_to_source(self.source.info)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        self.started_at = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="telemetry-loop")
        self._thread.start()
        log.info("engine started: %s, model=%s (%s), window=%ds",
                 self.source.info.detail, self.model.path.name,
                 self.model.feature_variant, self.window_sec)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        self.source.close()

    def reset(self) -> None:
        """Clear the books without restarting the poll loop."""
        with self._lock:
            self.samples.clear()
            self.predictions.clear()
            self.events = EventTracker(window_sec=self.window_sec,
                                       confidence=self.events.confidence)
            self.started_at = self.stream_now or time.time()
            self.exhausted = False

    # ── the loop ─────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                row = self.source.sample()
                if row is None:
                    self.exhausted = True
                    log.info("source exhausted")
                    return
                self._ingest(row)
                self.last_error = None
            except Exception as e:
                self.last_error = f"{type(e).__name__}: {e}"
                log.warning("sample/classify failed: %s", self.last_error)

            elapsed = time.monotonic() - t0
            self._stop.wait(max(0.0, self._interval - elapsed))

    def _ingest(self, row: dict) -> None:
        now = row.get("timestamp_epoch") or time.time()
        self.stream_now = now

        with self._lock:
            self.samples.append(row)
            # Replay traces carry their own label and re-declare it each
            # sample. A live GPU knows nothing, so the source must stay silent
            # -- otherwise it would overwrite the operator's mark every second
            # and the ledger would never count a single window.
            if self.source.provides_truth:
                self.events.declare(self.source.true_label(), t=now)
            elif self.truth_tracker is not None:
                try:
                    self.events.declare(self.truth_tracker.current_label(), t=now)
                except Exception as e:
                    # Losing ground truth must not stop the classifier.
                    log.warning("truth tracker failed: %s", e)
            self.events.observe_sample(now, float(row.get("gpu_utilization_pct") or 0.0))
            window = list(self.samples)[-self.window_sec:]

        if len(window) < MIN_WINDOW_SAMPLES:
            return

        t_a = time.perf_counter()
        frame = pd.DataFrame({c: [float(s.get(c) or 0.0) for s in window]
                              for c in SIGNALS + CODEC})
        feats = extract_features(frame)
        t_b = time.perf_counter()

        x = np.array([[float(feats.get(c, 0.0)) for c in self.model.feature_cols]],
                     dtype=np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        proba = self.model.model.predict_proba(x)[0]
        t_c = time.perf_counter()

        probs = {str(c): float(p) for c, p in zip(self.model.classes, proba)}
        top = max(probs, key=probs.get)

        compare = None
        if self.compare_model is not None:
            cm = self.compare_model
            xc = np.array([[float(feats.get(c, 0.0)) for c in cm.feature_cols]],
                          dtype=np.float32)
            xc = np.nan_to_num(xc, nan=0.0, posinf=0.0, neginf=0.0)
            cp = cm.model.predict_proba(xc)[0]
            cprobs = {str(c): float(v) for c, v in zip(cm.classes, cp)}
            compare = {"probs": cprobs, "predicted": max(cprobs, key=cprobs.get)}

        mid = None
        if self.model_id_model is not None:
            mm = self.model_id_model
            xm = np.array([[float(feats.get(c, 0.0)) for c in mm.feature_cols]],
                          dtype=np.float32)
            xm = np.nan_to_num(xm, nan=0.0, posinf=0.0, neginf=0.0)
            mp = mm.model.predict_proba(xm)[0]
            mprobs = {str(c): float(v) for c, v in zip(mm.classes, mp)}
            mid = self.model_id.observe(now, mprobs,
                                        self.events.declared_label, top)
            mid["probs"] = mprobs

        with self._lock:
            self.timing_ms = {"extract": round((t_b - t_a) * 1000, 2),
                              "predict": round((t_c - t_b) * 1000, 2)}
            alarm = self.events.observe_prediction(now, probs)
            self.predictions.append(dict(
                t=now, probs=probs, predicted=top,
                confidence=probs[top],
                window_samples=len(window),
                window_full=len(window) >= self.window_sec,
                false_alarm=alarm is not None,
                # Declared truth at the moment of this prediction. The timeline
                # draws it as a band under the probabilities, so the lag between
                # the truth flipping and the classifier following is read
                # straight off the picture.
                declared=self.events.declared_class,
                compare=compare,
                model_id=mid,
            ))

    # ── what the UI reads ────────────────────────────────────────────────────

    def sparklines(self) -> dict:
        with self._lock:
            recent = list(self.samples)[-SPARKLINE_SECONDS:]
        if not recent:
            return {"t": [], "signals": {}, "window_sec": self.window_sec}
        t0 = recent[0].get("timestamp_epoch", 0)
        return {
            "t": [round((s.get("timestamp_epoch", 0) - t0), 1) for s in recent],
            "n": len(recent),
            # How many trailing samples the classifier is currently reading, so
            # the UI can box exactly that region on every sparkline.
            "window_samples": min(self.window_sec, len(recent)),
            "window_sec": self.window_sec,
            "signals": {sig: [round(float(s.get(sig) or 0.0), 2) for s in recent]
                        for sig in RAW_SIGNALS},
        }

    @staticmethod
    def _agreement(preds: list[dict]) -> dict | None:
        """How often the two models call the same window the same way."""
        both = [p for p in preds if p.get("compare")]
        if not both:
            return None
        agree = sum(1 for p in both if p["compare"]["predicted"] == p["predicted"])
        return {
            "n": len(both),
            "agree": agree,
            "rate": round(agree / len(both), 4),
            "mean_conf_primary": round(
                sum(max(p["probs"].values()) for p in both) / len(both), 3),
            "mean_conf_compare": round(
                sum(max(p["compare"]["probs"].values()) for p in both) / len(both), 3),
        }

    def state(self) -> dict:
        now = self.stream_now if self.stream_now is not None else time.time()
        with self._lock:
            preds = list(self.predictions)
            latest = preds[-1] if preds else None
            n_samples = len(self.samples)
            timing = dict(self.timing_ms)
            events = self.events.to_dict(now)
            last_error = self.last_error
            exhausted = self.exhausted

        return {
            "now": now,
            "uptime_s": round(max(0.0, now - self.started_at), 1) if self.started_at else 0,
            "buffered_samples": n_samples,
            "window_sec": self.window_sec,
            "window_fill_frac": round(min(1.0, n_samples / self.window_sec), 3),
            "ready": n_samples >= MIN_WINDOW_SAMPLES,
            "latest": latest,
            "history": [dict(t=round(p["t"], 1), probs=p["probs"],
                             predicted=p["predicted"],
                             window_full=p["window_full"],
                             false_alarm=p["false_alarm"],
                             declared=p["declared"],
                             compare=p["compare"])
                        for p in preds[-HISTORY_SECONDS:]],
            "classes": self.model.classes,
            "events": events,
            "timing_ms": timing,
            "source": self.source.info.to_dict(),
            "truth_source": ("trace" if self.source.provides_truth
                             else "collection" if self.truth_tracker is not None
                             else "operator"),
            "compare_agreement": self._agreement(preds),
            "model_id": self.model_id.to_dict() if self.model_id else None,
            "exhausted": exhausted,
            "last_error": last_error,
            "sparklines": self.sparklines(),
        }
