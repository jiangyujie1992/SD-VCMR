from __future__ import annotations

import json
from pathlib import Path

import pytest

from mtvr.config import load_experiment_config


def test_release_config_is_strict_and_candidate_protocol_is_consistent() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(root / "configs" / "tvr_mml.json")
    assert config.schema_version == 1
    assert config.train.batch_size >= config.train.candidate_sample_size
    assert config.train.candidate_sample_size == 100
    assert config.train.candidate_keep == 10
    assert config.evaluation.ks == (1, 10, 100)
    assert len(config.experiment.seeds) == 3


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    payload = json.loads((root / "configs" / "tvr_mml.json").read_text())
    payload["train"]["unused_typo"] = 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="unused_typo"):
        load_experiment_config(path)
