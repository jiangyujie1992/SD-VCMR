from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtvr.config import load_experiment_config
from mtvr.data import read_jsonl
from mtvr.metrics import aggregate_run_metrics, evaluate_predictions
from mtvr.runtime import atomic_write_json, sha256_file, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one run or aggregate multiple seeded MTVR runs"
    )
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, help="use evaluation ks/ious from config")
    parser.add_argument("--ks", type=int, nargs="+", help="override Recall cutoffs")
    parser.add_argument("--ious", type=float, nargs="+", help="override IoU thresholds")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.config:
        evaluation = load_experiment_config(args.config).evaluation
        ks = tuple(args.ks) if args.ks else evaluation.ks
        ious = tuple(args.ious) if args.ious else evaluation.ious
    else:
        ks = tuple(args.ks) if args.ks else (1, 10, 100)
        ious = tuple(args.ious) if args.ious else (0.5, 0.7)

    paths = [path.resolve() for path in args.predictions]
    results = [
        evaluate_predictions(read_jsonl(path), ks=ks, ious=ious) for path in paths
    ]
    evaluation_result = results[0] if len(results) == 1 else aggregate_run_metrics(results)
    payload = {
        "created_at_utc": utc_now(),
        "prediction_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
        "evaluation": evaluation_result,
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    print(text, end="")
    if args.output:
        atomic_write_json(args.output.resolve(), payload)


if __name__ == "__main__":
    main()
