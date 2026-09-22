"""Model identification: which LLM is running, from telemetry alone.

A different question from the workload classifier's, and a more delicate one to
display honestly, because this classifier is **closed-set**: it was fitted on
ten specific LLMs and its softmax always sums to one over exactly those ten. It
has no "none of these" and no abstain. Point it at an idle GPU, a ResNet, or an
eleventh LLM and it will still name one of its ten, often confidently.

So the panel it feeds must answer two questions separately:

  1. What does the classifier say?
  2. Is that answer meaningful right now?

(2) is not something the classifier can tell you, so this module tracks it
from outside: whether the workload currently looks like LLM inference at all,
and whether the serving configuration matches the one the corpus was collected
under. The model-ID corpus is a single configuration -- batch 4, 256-token
prompts, 64 new tokens -- and its own documentation predicts the classifier
degrades when that varies.
"""
from __future__ import annotations

from collections import deque


def model_class_for_label(label: str | None, classes: list[str]) -> str | None:
    """Map a workload label onto one of the classifier's model classes.

    Corpus labels are `llm_infer_<model>`; the batch sweep uses
    `llm_infer_sweep_<model>_bs<N>`, where <model> is a short prefix of the
    class name (pythia -> pythia_1p4b). Prefix matching covers both, and
    returns None for anything that is not an LLM inference workload at all --
    which is the answer that keeps the panel honest.
    """
    if not label:
        return None
    stem = label
    for prefix in ("llm_infer_sweep_", "llm_infer_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
            break
    else:
        return None

    # Drop a trailing batch-size token: pythia_bs16 -> pythia
    parts = stem.split("_")
    if parts and parts[-1].startswith("bs") and parts[-1][2:].isdigit():
        parts = parts[:-1]
    stem = "_".join(parts)

    if stem in classes:
        return stem
    for c in classes:
        if c.startswith(stem + "_") or c == stem:
            return c
    return None


def batch_size_for_label(label: str | None) -> int | None:
    """Serving batch size, when the workload label carries one."""
    if not label:
        return None
    for tok in reversed(label.split("_")):
        if tok.startswith("bs") and tok[2:].isdigit():
            return int(tok[2:])
    return None


class ModelIdTracker:
    """Scores model-ID predictions, but only where the answer can mean anything."""

    def __init__(self, classes: list[str], trained_batch_size: int | None = 4,
                 history: int = 900):
        self.classes = classes
        self.trained_batch_size = trained_batch_size
        self.correct = 0
        self.scored = 0
        # Per-batch-size accuracy: the whole point of the batch sweep is that
        # the classifier was fitted at one configuration, so a single pooled
        # accuracy would hide exactly the effect worth seeing.
        self.by_batch: dict[int | str, dict] = {}
        self.recent: deque[dict] = deque(maxlen=history)

    def observe(self, t: float, probs: dict, declared_label: str | None,
                workload_class: str | None) -> dict:
        top = max(probs, key=probs.get)
        truth = model_class_for_label(declared_label, self.classes)
        batch = batch_size_for_label(declared_label)

        # Applicable only when the thing running really is LLM inference. With
        # no declared truth we fall back to the workload classifier's opinion,
        # and say that is what we are relying on.
        if truth is not None:
            applicable, basis = True, "declared"
        elif declared_label is not None:
            applicable, basis = False, "declared"
        else:
            applicable = workload_class == "ml_inference"
            basis = "workload_classifier"

        entry = {
            "t": t, "predicted": top, "confidence": probs[top],
            "truth": truth, "correct": (truth is not None and top == truth),
            "batch_size": batch, "applicable": applicable, "basis": basis,
            "off_distribution": (batch is not None
                                 and self.trained_batch_size is not None
                                 and batch != self.trained_batch_size),
        }
        self.recent.append(entry)

        if truth is not None:
            self.scored += 1
            self.correct += entry["correct"]
            key = batch if batch is not None else "unknown"
            b = self.by_batch.setdefault(key, {"scored": 0, "correct": 0})
            b["scored"] += 1
            b["correct"] += entry["correct"]

        return entry

    def to_dict(self) -> dict:
        by_batch = {
            str(k): {**v, "accuracy": round(v["correct"] / v["scored"], 4)}
            for k, v in sorted(self.by_batch.items(), key=lambda kv: str(kv[0]))
            if v["scored"]
        }
        return {
            "scored": self.scored,
            "correct": self.correct,
            "accuracy": round(self.correct / self.scored, 4) if self.scored else None,
            "by_batch": by_batch,
            "trained_batch_size": self.trained_batch_size,
            "classes": self.classes,
        }
