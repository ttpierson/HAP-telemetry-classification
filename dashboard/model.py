"""Loading a classifier bundle, and checking it against the machine it runs on.

Bundles come from train_classifier.py: a joblib with the model and its exact
feature columns, plus metadata.json beside it with the training corpus, its
power regime and the grouped-CV scores.

Provenance matters because a model trained under one power regime reads every
workload under another as something it has never seen. That is a property of
the deployment, not of the model file, so it is checked at load time against
the actual GPU (or, for replay, against the trace's peak power).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from gputel.features import CODEC, SIGNALS, extract_features

FEATURE_VARIANT = "window_local"


class FeatureContractError(RuntimeError):
    """The live path cannot supply what the model was trained on."""


@dataclass
class LoadedModel:
    task: str
    model: object
    feature_cols: list[str]
    classes: list[str]
    window_sec: int
    stride_sec: int
    metadata: dict
    path: Path
    # Bundle keys beyond the ones modelled explicitly (memory_features_removed, ...)
    extras: dict = field(default_factory=dict)
    feature_variant: str = FEATURE_VARIANT

    # ── provenance ───────────────────────────────────────────────────────────

    @property
    def provenance(self) -> dict:
        p = dict(self.metadata.get("provenance") or {})
        names = p.get("gpu_names") or []
        p.setdefault("gpu_name", names[0] if len(names) == 1 else None)
        return p

    @property
    def trained_power_w(self) -> float | None:
        """The power regime of the training corpus.

        The recorded enforced power limit when traces carry one (the most
        common value, so one odd GPU cannot skew it); otherwise the corpus's
        peak observed draw, which sits at the cap for any capped workload.
        """
        limits = self.provenance.get("power_limits_w") or []
        if limits:
            return float(max(set(limits), key=limits.count))
        return self.provenance.get("peak_power_w_max")

    def compare_to_source(self, source_info) -> list[dict]:
        """Warnings about running this model on this source. Returned, not
        raised: a mismatched model is still worth showing, in red."""
        warnings = []
        trained_gpu = self.provenance.get("gpu_name")
        trained_w = self.trained_power_w
        if trained_gpu and source_info.gpu_name and trained_gpu != source_info.gpu_name:
            warnings.append(dict(level="error", field="gpu",
                                 message=f"model trained on {trained_gpu}, this is "
                                         f"{source_info.gpu_name}"))

        if source_info.kind == "replay":
            peak = source_info.observed_max_power_w
            if trained_w and peak and not (0.75 * trained_w <= peak <= 1.15 * trained_w):
                warnings.append(dict(
                    level="error", field="power",
                    message=f"model trained at ~{trained_w:.0f} W, but this trace peaks at "
                            f"{peak:.0f} W under load — it was almost certainly "
                            f"collected under a different power regime. Predictions "
                            f"are miscalibrated."))
            # A model scoring a run it was fitted on looks flawless and none of
            # it generalises.
            trained_runs = set(self.metadata.get("train_run_ids") or [])
            seen = [r for r in source_info.run_ids if r in trained_runs]
            if seen:
                warnings.append(dict(
                    level="error", field="corpus",
                    message=f"{len(seen)} of {len(source_info.run_ids)} replayed trace(s) "
                            f"are IN this model's training corpus: detection will look "
                            f"instant and false alarms near zero. Train with "
                            f"--holdout-config and replay a held-out config instead."))
            elif source_info.run_ids and not trained_runs:
                warnings.append(dict(level="warn", field="corpus",
                                     message="bundle does not record its training runs, "
                                             "so in-sample replay cannot be detected."))
            return warnings

        live_w = source_info.power_limit_w
        if not trained_w:
            warnings.append(dict(level="warn", field="power",
                                 message="bundle records no training power regime; it "
                                         "cannot be checked against this GPU."))
        elif live_w and abs(trained_w - live_w) > 0.1 * live_w:
            warnings.append(dict(
                level="error", field="power",
                message=f"model trained at ~{trained_w:.0f} W, this GPU is capped at "
                        f"{live_w:.0f} W — every workload here will look unlike the "
                        f"training data. Predictions are systematically miscalibrated."))
        return warnings

    # ── headline accuracy ────────────────────────────────────────────────────

    def scores(self) -> dict:
        s = (self.metadata.get("scores") or {}).get(self.task) or {}
        return {"grouped_by_workload": s.get("grouped_by_workload"),
                "grouped_by_run": s.get("grouped_by_run")}

    def binary_operating_point(self) -> dict | None:
        """Recall and false-positive rate for ml_training vs everything else,
        from this model's own held-out (grouped-by-workload) confusion matrix."""
        all_scores = self.metadata.get("scores") or {}
        for task in ("binary", "threeway"):
            s = (all_scores.get(task) or {}).get("grouped_by_workload")
            if not s or not s.get("confusion") or "ml_training" not in s["labels"]:
                continue
            cm = np.array(s["confusion"], dtype=float)
            i = s["labels"].index("ml_training")
            tp = cm[i, i]
            fn = cm[i].sum() - tp
            fp = cm[:, i].sum() - tp
            tn = cm.sum() - tp - fn - fp
            if (tp + fn) and (fp + tn):
                return {"source_task": task, "recall": float(tp / (tp + fn)),
                        "fpr": float(fp / (fp + tn)), "folds": s.get("folds")}
        return None


def precision_at_base_rate(recall: float, fpr: float, base_rate: float) -> float:
    """P(training | flagged) once you account for how rare training is."""
    tp, fp = base_rate * recall, (1.0 - base_rate) * fpr
    return float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0


def live_feature_names() -> set[str]:
    """Features the live path produces, found by running the real extractor."""
    probe = pd.DataFrame({c: np.linspace(1.0, 2.0, 30) for c in SIGNALS + CODEC})
    return set(extract_features(probe))


def load_model(path: str | Path) -> LoadedModel:
    path = Path(path)
    bundle = joblib.load(path)
    meta_path = path.parent / "metadata.json"
    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    cols = list(bundle["feature_cols"])
    missing = [c for c in cols if c not in live_feature_names()]
    if missing:
        raise FeatureContractError(
            f"{path.name} expects {len(missing)} feature(s) the live path cannot "
            f"compute: {', '.join(missing)}. Retrain it with train_classifier.py. "
            "Zero-filling them would be a silent train/serve mismatch.")

    known = {"model", "feature_cols", "classes", "window_sec", "stride_sec", "task"}
    return LoadedModel(
        task=bundle.get("task", "threeway"), model=bundle["model"], feature_cols=cols,
        classes=[str(c) for c in bundle["classes"]],
        window_sec=int(bundle["window_sec"]), stride_sec=int(bundle["stride_sec"]),
        metadata=metadata, path=path,
        extras={k: v for k, v in bundle.items() if k not in known})
