"""Score a JSONL file of ranked retrieval predictions."""

import argparse
import json
from pathlib import Path

from retrieval_metrics import evaluate_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path, help="one query per JSONL row")
    args = parser.parse_args()
    with args.predictions.open(encoding="utf-8") as stream:
        records = (json.loads(line) for line in stream if line.strip())
        result = evaluate_predictions(records)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
