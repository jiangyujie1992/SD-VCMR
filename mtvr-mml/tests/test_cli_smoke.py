from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def test_end_to_end_cli_smoke(
    tmp_path: Path, run_cli: Callable[..., None]
) -> None:
    data = tmp_path / "data"
    train = tmp_path / "train"
    predictions = tmp_path / "predictions.jsonl"
    metrics = tmp_path / "metrics.json"
    run_cli("scripts/make_toy_data.py", "--output", str(data), "--videos", "4")
    run_cli("scripts/validate_data.py", "--config", str(data / "config.json"))
    run_cli(
        "scripts/train.py",
        "--config",
        str(data / "config.json"),
        "--output",
        str(train),
        "--device",
        "cpu",
        "--max-steps",
        "1",
    )
    run_cli(
        "scripts/infer.py",
        "--config",
        str(data / "config.json"),
        "--checkpoint",
        str(train / "checkpoint_last.pt"),
        "--output",
        str(predictions),
        "--device",
        "cpu",
        "--max-queries",
        "2",
    )
    run_cli(
        "scripts/evaluate.py",
        "--config",
        str(data / "config.json"),
        "--predictions",
        str(predictions),
        "--output",
        str(metrics),
    )
    assert predictions.is_file()
    assert predictions.with_suffix(".jsonl.meta.json").is_file()
    checkpoint = train / "checkpoint_last.pt"
    assert checkpoint.is_file()
    metadata = json.loads((train / "run_metadata.json").read_text())
    assert metadata["status"] == "completed"
    assert metadata["candidate_protocol"]["ground_truth_forced"] is True
    result = json.loads(metrics.read_text())
    assert result["evaluation"]["queries"] == 2
