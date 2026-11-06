from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtvr.config import load_experiment_config
from mtvr.data import validate_manifests
from mtvr.runtime import atomic_write_json, utc_now


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate MTVR manifests, IDs, labels, and every NumPy feature"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "all"), default="all")
    parser.add_argument("--output", type=Path, help="optional JSON validation report")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    videos = config.resolve_path(config.data.videos)
    splits = ("train", "val") if args.split == "all" else (args.split,)
    reports = {}
    for split in splits:
        query_value = (
            config.data.train_queries if split == "train" else config.data.val_queries
        )
        reports[split] = validate_manifests(
            config.resolve_path(query_value),
            videos,
            model_config=config.model,
            expected_proxy_styles=config.data.expected_proxy_styles,
            require_labels=True,
        )
    payload = {
        "status": "pass",
        "validated_at_utc": utc_now(),
        "config": str(config.config_path),
        "splits": reports,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.output:
        atomic_write_json(args.output.resolve(), payload)


if __name__ == "__main__":
    main()
