from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtvr.runtime import atomic_write_json, sha256_file, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic data -> train -> infer -> evaluate smoke test"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def run(arguments: list[str]) -> None:
    print("+ " + " ".join(arguments), flush=True)
    subprocess.run(arguments, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    if args.max_steps <= 0:
        raise ValueError("max_steps must be positive")
    output = args.output.resolve()
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"smoke output exists: {output}; pass --overwrite to replace it"
            )
        if output in {ROOT, ROOT.parent, Path(output.anchor)}:
            raise ValueError(f"refusing to remove unsafe smoke output path: {output}")
        shutil.rmtree(output)
    data = output / "data"
    train = output / "train"
    predictions = output / "predictions.jsonl"
    metrics = output / "metrics.json"
    python = sys.executable

    run(
        [
            python,
            "scripts/make_toy_data.py",
            "--output",
            str(data),
            "--seed",
            str(args.seed),
        ]
    )
    run(
        [
            python,
            "scripts/validate_data.py",
            "--config",
            str(data / "config.json"),
        ]
    )
    run(
        [
            python,
            "scripts/train.py",
            "--config",
            str(data / "config.json"),
            "--output",
            str(train),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
            "--max-steps",
            str(args.max_steps),
        ]
    )
    run(
        [
            python,
            "scripts/infer.py",
            "--config",
            str(data / "config.json"),
            "--checkpoint",
            str(train / "checkpoint_last.pt"),
            "--split",
            "val",
            "--output",
            str(predictions),
            "--device",
            args.device,
        ]
    )
    run(
        [
            python,
            "scripts/evaluate.py",
            "--config",
            str(data / "config.json"),
            "--predictions",
            str(predictions),
            "--output",
            str(metrics),
        ]
    )
    summary = {
        "status": "pass",
        "completed_at_utc": utc_now(),
        "purpose": "functional smoke test; metrics are not paper results",
        "seed": args.seed,
        "max_steps": args.max_steps,
        "artifacts": {
            "checkpoint": str(train / "checkpoint_last.pt"),
            "checkpoint_sha256": sha256_file(train / "checkpoint_last.pt"),
            "predictions": str(predictions),
            "predictions_sha256": sha256_file(predictions),
            "metrics": str(metrics),
            "metrics_sha256": sha256_file(metrics),
        },
    }
    atomic_write_json(output / "SMOKE_TEST_REPORT.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
