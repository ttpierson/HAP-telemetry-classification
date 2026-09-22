"""The three things worth watching live, and the bookkeeping behind them.

  1. Detection latency  — a workload starts; how many seconds until the
                          classifier is confident? Measured from a signal-based
                          onset trigger, deliberately independent of the
                          classifier so it cannot mark its own homework.
  2. Transitions        — inference -> training mid-stream. Lag, confusion
                          during the changeover, overshoot afterwards.
  3. False positives    — spurious ml_training windows arriving in real time
                          while the GPU is known to be doing something else.

Ground truth for 2 and 3 comes from the replay trace's own labels, or from an
operator marking the state live. With neither, the panels say so rather than
inventing a baseline.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from gputel.features import threeway_label

TRAINING = "ml_training"

# Onset trigger. Deliberately crude and classifier-independent: utilisation
# crossing up and staying there. If the trigger used the model's own output,
# "time from onset to confident" would be circular.
ONSET_UTIL_PCT = 30.0
ONSET_SUSTAIN_S = 3
IDLE_UTIL_PCT = 5.0
IDLE_SUSTAIN_S = 5

# A call counts as confident once probability holds above the threshold for a
# few consecutive seconds -- one lucky second is not a detection.
DEFAULT_CONFIDENCE = 0.80
CONFIDENT_SUSTAIN_S = 3


def to_threeway(label: str | None) -> str | None:
    return threeway_label(label) if label else None


@dataclass
class Episode:
    """One workload onset or transition, and the classifier's response to it."""
    t0: float
    kind: str                              # "onset" | "transition" | "mark"
    true_label: str | None = None          # workload label, if known
    true_class: str | None = None          # three-way class, if known
    from_class: str | None = None          # what it was before (transitions)
    window_sec: int = 30
    # True when this "onset" is really just monitoring starting on a workload
    # that was already running. The window was full of it from the first
    # sample, so any latency measured here is the buffer filling, not detection.
    at_monitor_start: bool = False

    # (elapsed_s, {class: prob}, predicted_class)
    trajectory: list[tuple[float, dict, str]] = field(default_factory=list)

    time_to_confident: float | None = None   # first sustained call, any class
    confident_class: str | None = None
    time_to_correct: float | None = None     # first sustained CORRECT call
    settled: bool = False

    _streak_class: str | None = None
    _streak_start: float | None = None
    _correct_streak_start: float | None = None

    def observe(self, t: float, probs: dict, threshold: float) -> None:
        elapsed = t - self.t0
        top = max(probs, key=probs.get)
        self.trajectory.append((round(elapsed, 1), probs, top))

        confident = probs[top] >= threshold

        # Sustained confident call in any class -> detection latency.
        if confident and top == self._streak_class:
            if (self.time_to_confident is None and self._streak_start is not None
                    and t - self._streak_start >= CONFIDENT_SUSTAIN_S - 1):
                self.time_to_confident = round(self._streak_start - self.t0, 1)
                self.confident_class = top
        elif confident:
            self._streak_class, self._streak_start = top, t
        else:
            self._streak_class, self._streak_start = None, None

        # Sustained confident call in the RIGHT class -> the honest latency.
        if self.true_class:
            right = confident and top == self.true_class
            if right and self._correct_streak_start is not None:
                if (self.time_to_correct is None
                        and t - self._correct_streak_start >= CONFIDENT_SUSTAIN_S - 1):
                    self.time_to_correct = round(
                        self._correct_streak_start - self.t0, 1)
                    self.settled = True
            elif right:
                self._correct_streak_start = t
            else:
                self._correct_streak_start = None

    def window_fill(self, t: float) -> float:
        """Fraction of the 30 s window that postdates t0.

        For the first 30 seconds the classifier is reading a window still part
        filled with whatever came before. Any latency number that ignores this
        is really measuring the buffer draining, not the model deciding.
        """
        return min(1.0, max(0.0, (t - self.t0) / self.window_sec))

    def to_dict(self, now: float) -> dict:
        return dict(
            t0=self.t0, kind=self.kind, age_s=round(now - self.t0, 1),
            true_label=self.true_label, true_class=self.true_class,
            from_class=self.from_class,
            window_fill=round(self.window_fill(now), 3),
            time_to_confident=self.time_to_confident,
            confident_class=self.confident_class,
            time_to_correct=self.time_to_correct,
            settled=self.settled,
            at_monitor_start=self.at_monitor_start,
            trajectory=self.trajectory[-240:],
        )


@dataclass
class FalseAlarm:
    t: float
    confidence: float
    true_class: str
    true_label: str | None


class EventTracker:
    """Watches the sample stream and the prediction stream, and keeps the books."""

    def __init__(self, window_sec: int = 30, confidence: float = DEFAULT_CONFIDENCE):
        self.window_sec = window_sec
        self.confidence = confidence

        self.episodes: deque[Episode] = deque(maxlen=40)
        self.current: Episode | None = None

        self._active = False
        self._active_since: float | None = None
        self._idle_since: float | None = None

        self._declared_label: str | None = None    # from replay or operator
        self._declared_class: str | None = None
        self._monitor_start: float | None = None
        # Whether the GPU was ALREADY busy the moment we started watching. That
        # -- not "an onset happened early" -- is what makes a latency
        # unmeasurable, because it means the workload predates the buffer.
        self._busy_at_monitor_start = False

        # False-positive ledger, counted only while truth is known not-training.
        self.false_alarms: deque[FalseAlarm] = deque(maxlen=500)
        self.windows_known_not_training = 0
        self.windows_known_training = 0
        self.missed_detections = 0

    # ── ground truth ─────────────────────────────────────────────────────────

    def declare(self, label: str | None, t: float | None = None) -> None:
        """Set ground truth: replay trace label, or an operator's mark.

        A change of declared class is itself a transition worth timing.
        """
        t = t if t is not None else time.time()
        new_class = to_threeway(label)
        if label == self._declared_label:
            return

        prev_class = self._declared_class
        self._declared_label = label
        self._declared_class = new_class

        if label is None:
            return

        # Only open a transition episode if a workload was already running --
        # otherwise the onset detector owns this moment.
        if prev_class is not None and prev_class != new_class:
            self._open(Episode(t0=t, kind="transition", true_label=label,
                               true_class=new_class, from_class=prev_class,
                               window_sec=self.window_sec))
        elif self.current is not None and self.current.true_class is None:
            # Operator labelled a workload the onset detector already caught.
            self.current.true_label = label
            self.current.true_class = new_class
        # A first declaration with nothing running is a baseline, not an event.
        # It establishes what "not training" means for the ledger; the onset
        # detector will time the workload when it actually starts.

    @property
    def declared_class(self) -> str | None:
        return self._declared_class

    @property
    def declared_label(self) -> str | None:
        return self._declared_label

    # ── the sample stream ────────────────────────────────────────────────────

    def observe_sample(self, t: float, util: float) -> None:
        """Signal-based onset and offset detection, classifier-independent."""
        if self._monitor_start is None:
            self._monitor_start = t
            self._busy_at_monitor_start = util >= ONSET_UTIL_PCT
        if util >= ONSET_UTIL_PCT:
            self._idle_since = None
            if self._active_since is None:
                self._active_since = t
            if not self._active and t - self._active_since >= ONSET_SUSTAIN_S - 1:
                self._active = True
                self._open(Episode(
                    t0=self._active_since, kind="onset",
                    true_label=self._declared_label,
                    true_class=self._declared_class,
                    window_sec=self.window_sec))
        elif util < IDLE_UTIL_PCT:
            self._active_since = None
            if self._idle_since is None:
                self._idle_since = t
            if self._active and t - self._idle_since >= IDLE_SUSTAIN_S - 1:
                self._active = False
                # Same reasoning: a transition mid-measurement survives the
                # GPU going quiet, because going quiet is often part of it.
                if not (self.current and self.current.kind == "transition"
                        and not self.current.settled):
                    self.current = None
        else:
            self._active_since = None

    # ── the prediction stream ────────────────────────────────────────────────

    def observe_prediction(self, t: float, probs: dict) -> dict | None:
        """Feed one classification in. Returns a false alarm if this was one."""
        if self.current is not None:
            self.current.observe(t, probs, self.confidence)

        top = max(probs, key=probs.get)
        alarm = None

        if self._declared_class is not None:
            if self._declared_class == TRAINING:
                self.windows_known_training += 1
                if top != TRAINING:
                    self.missed_detections += 1
            else:
                self.windows_known_not_training += 1
                if top == TRAINING:
                    alarm = FalseAlarm(t=t, confidence=probs[top],
                                       true_class=self._declared_class,
                                       true_label=self._declared_label)
                    self.false_alarms.append(alarm)

        return alarm.__dict__ if alarm else None

    def _open(self, ep: Episode) -> None:
        """Start an episode, merging with one already in flight.

        A workload change shows up twice within a second or two: as a
        utilisation edge and as a change of declared label. They are one event.
        Merging keeps the earlier timestamp — the first evidence of the change
        — and the richer labelling, instead of restarting the latency clock
        halfway through measuring it.
        """
        cur = self.current
        # A transition still waiting to be resolved owns the clock. A training
        # run can idle for a minute in model init and data loading before
        # utilisation climbs, and that startup is part of the transition, not a
        # separate event -- restarting the clock at the utilisation edge would
        # quietly discount the slowest and most interesting part of the lag.
        if (cur is not None and ep.kind == "onset"
                and cur.kind == "transition" and not cur.settled):
            return

        if cur is not None and abs(ep.t0 - cur.t0) < self.window_sec:
            if ep.t0 < cur.t0:
                cur.t0 = ep.t0
            if ep.true_class and not cur.true_class:
                cur.true_label, cur.true_class = ep.true_label, ep.true_class
            if ep.kind == "transition":
                cur.kind = "transition"
                cur.from_class = ep.from_class or cur.from_class
                # A declared switch overrides whatever label the episode
                # inherited from before the switch.
                cur.true_label, cur.true_class = ep.true_label, ep.true_class
                cur.time_to_confident = None
                cur.confident_class = None
                cur.time_to_correct = None
                cur.settled = False
                cur._streak_class = cur._streak_start = None
                cur._correct_streak_start = None
                cur.trajectory.clear()
            return

        # An onset cannot be timed only when the workload predates monitoring:
        # the GPU was already busy when we started watching, so there is no
        # "before" in the buffer to measure against. An onset that happens soon
        # after monitoring starts on an IDLE GPU is perfectly measurable -- the
        # window legitimately holds the idle period the workload interrupted,
        # which is exactly what the latency is measured across. Suppressing
        # those too would discard the normal case of starting the dashboard and
        # then starting a workload.
        if (self._busy_at_monitor_start and self._monitor_start is not None
                and ep.t0 - self._monitor_start < self.window_sec):
            ep.at_monitor_start = True

        self.current = ep
        self.episodes.append(ep)

    # ── reporting ────────────────────────────────────────────────────────────

    def ledger(self, now: float) -> dict:
        n_fp = len(self.false_alarms)
        observed = self.windows_known_not_training
        recent = [a for a in self.false_alarms if now - a.t <= 300]
        return dict(
            windows_known_not_training=observed,
            windows_known_training=self.windows_known_training,
            false_alarms=n_fp,
            missed_detections=self.missed_detections,
            observed_fp_rate=round(n_fp / observed, 4) if observed else None,
            observed_miss_rate=(
                round(self.missed_detections / self.windows_known_training, 4)
                if self.windows_known_training else None),
            false_alarms_per_min_5m=round(len(recent) / 5.0, 2) if recent else 0.0,
            recent=[dict(t=a.t, age_s=round(now - a.t, 1),
                         confidence=round(a.confidence, 3),
                         true_class=a.true_class, true_label=a.true_label)
                    for a in list(self.false_alarms)[-25:][::-1]],
        )

    def to_dict(self, now: float) -> dict:
        return dict(
            declared_label=self._declared_label,
            declared_class=self._declared_class,
            gpu_active=self._active,
            current=self.current.to_dict(now) if self.current else None,
            recent_episodes=[e.to_dict(now) for e in list(self.episodes)[-6:][::-1]],
            ledger=self.ledger(now),
            confidence_threshold=self.confidence,
        )
