"""Live GPU workload classifier dashboard.

    # live, on the machine with the GPU
    python -m dashboard --source nvml --gpu 0 --model models/workload_200w/threeway.joblib

    # offline replay, no GPU required
    python -m dashboard --source replay --model models/holdout/threeway.joblib \
        --trace a.parquet --trace b.parquet --speed 10

Open http://127.0.0.1:8000. Model bundles come from train_classifier.py.
"""
from __future__ import annotations

import argparse
import logging
import sys

from .engine import Engine
from .model import FeatureContractError, load_model
from .server import create_app


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m dashboard", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["nvml", "replay"], default="replay")
    ap.add_argument("--gpu", type=int, default=None,
                    help="NVML index to monitor (nvml source). Explicit only: this "
                         "tool never enumerates every device.")
    ap.add_argument("--trace", action="append", default=[],
                    help="parquet trace to replay; repeat to chain traces and "
                         "synthesise a workload transition")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="replay speed multiplier (1.0 = real time)")
    ap.add_argument("--loop", action="store_true", help="loop the replay")
    ap.add_argument("--model", required=True, help="a workload .joblib bundle")
    ap.add_argument("--compare-model", default=None,
                    help="a second bundle scoring the identical window, side by side. "
                         "Pair a model that trained on this workload with one that "
                         "never saw it to see what headline accuracy leaves out.")
    ap.add_argument("--model-id-model", default=None,
                    help="a model-ID bundle (which LLM is running). Closed-set: it "
                         "always names one of its trained models, so the panel "
                         "states when its answer is meaningful.")
    ap.add_argument("--model-id-batch-size", type=int, default=4,
                    help="serving batch size the model-ID corpus was collected at; "
                         "windows at other batch sizes are marked off-distribution")
    ap.add_argument("--track-collection", action="store_true",
                    help="take ground truth from a running collect_telemetry.py on "
                         "this GPU instead of the operator's dropdown (nvml, Linux)")
    ap.add_argument("--confidence", type=float, default=0.80,
                    help="probability a call must hold to count as confident")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address. There is no authentication; keep loopback "
                         "unless you mean to expose it.")
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, datefmt="%H:%M:%S",
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    def load(path, what):
        try:
            return load_model(path)
        except FeatureContractError as e:
            print(f"\nERROR in {what}: {e}\n", file=sys.stderr)
            raise SystemExit(2) from e

    model = load(a.model, "--model")
    compare = load(a.compare_model, "--compare-model") if a.compare_model else None
    model_id = load(a.model_id_model, "--model-id-model") if a.model_id_model else None

    if a.source == "nvml":
        if a.gpu is None:
            ap.error("--source nvml needs --gpu")
        from .sources import NvmlSource
        source = NvmlSource(a.gpu)
    else:
        if not a.trace:
            ap.error("--source replay needs at least one --trace")
        from .sources import ReplaySource
        source = ReplaySource(a.trace, speed=a.speed, loop=a.loop)

    tracker = None
    if a.track_collection:
        if a.source != "nvml":
            ap.error("--track-collection only applies to --source nvml")
        from .collection import CollectionTracker
        tracker = CollectionTracker(gpu_index=a.gpu)

    engine = Engine(source, model, confidence=a.confidence, compare_model=compare,
                    truth_tracker=tracker, model_id_model=model_id,
                    model_id_batch_size=a.model_id_batch_size)
    for w in engine.provenance_warnings:
        logging.warning("PROVENANCE [%s] %s", w["level"], w["message"])

    import uvicorn
    uvicorn.run(create_app(engine), host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
