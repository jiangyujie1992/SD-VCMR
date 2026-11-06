from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from mtvr.data import MTVRDataset, collate_samples, load_feature


def write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )


def test_dataset_and_padding(tmp_path: Path) -> None:
    np.save(tmp_path / "v0.npy", np.ones((5, 6), dtype=np.float32))
    np.save(tmp_path / "s0.npy", np.ones((5, 4), dtype=np.float32))
    np.save(tmp_path / "v1.npy", np.ones((3, 6), dtype=np.float32))
    np.save(tmp_path / "s1.npy", np.ones((3, 4), dtype=np.float32))
    np.save(tmp_path / "q0.npy", np.ones(4, dtype=np.float32))
    np.save(tmp_path / "q1.npy", np.ones(4, dtype=np.float32))
    np.save(tmp_path / "p0.npy", np.ones((3, 7), dtype=np.float32))
    np.save(tmp_path / "p1.npy", np.ones((3, 7), dtype=np.float32))
    videos = tmp_path / "videos.jsonl"
    queries = tmp_path / "queries.jsonl"
    write_jsonl(
        videos,
        [
            {
                "video_id": "v0",
                "video_feature": "v0.npy",
                "subtitle_feature": "s0.npy",
            },
            {
                "video_id": "v1",
                "video_feature": "v1.npy",
                "subtitle_feature": "s1.npy",
            },
        ],
    )
    write_jsonl(
        queries,
        [
            {
                "query_id": "q0",
                "video_id": "v0",
                "query_feature": "q0.npy",
                "proxy_feature": "p0.npy",
                "start_clip": 1,
                "end_clip": 3,
            },
            {
                "query_id": "q1",
                "video_id": "v1",
                "query_feature": "q1.npy",
                "proxy_feature": "p1.npy",
                "start_clip": 0,
                "end_clip": 2,
            },
        ],
    )
    dataset = MTVRDataset(queries, videos)
    batch = collate_samples([dataset[0], dataset[1]])
    assert batch["video_features"].shape == (2, 5, 6)
    assert batch["subtitle_features"].shape == (2, 5, 4)
    assert batch["query_features"].shape == (2, 4)
    assert batch["proxy_features"].shape == (2, 3, 7)
    assert batch["clip_mask"].sum(dim=1).tolist() == [5, 3]
    assert batch["start_clip"].tolist() == [1, 0]


def test_feature_validation_rejects_nonfinite_and_wrong_rank(tmp_path: Path) -> None:
    nonfinite = tmp_path / "nonfinite.npy"
    wrong_rank = tmp_path / "wrong_rank.npy"
    np.save(nonfinite, np.asarray([1.0, np.nan], dtype=np.float32))
    np.save(wrong_rank, np.ones((2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="non-finite"):
        load_feature(nonfinite, expected_ndim=1)
    with pytest.raises(ValueError, match="expected 1 dimensions"):
        load_feature(wrong_rank, expected_ndim=1)


def test_dataset_rejects_duplicate_query_ids(tmp_path: Path) -> None:
    np.save(tmp_path / "v.npy", np.ones((2, 3), dtype=np.float32))
    np.save(tmp_path / "s.npy", np.ones((2, 3), dtype=np.float32))
    np.save(tmp_path / "q.npy", np.ones(3, dtype=np.float32))
    np.save(tmp_path / "p.npy", np.ones((3, 3), dtype=np.float32))
    videos = tmp_path / "videos.jsonl"
    queries = tmp_path / "queries.jsonl"
    write_jsonl(
        videos,
        [{"video_id": "v", "video_feature": "v.npy", "subtitle_feature": "s.npy"}],
    )
    query = {
        "query_id": "duplicate",
        "video_id": "v",
        "query_feature": "q.npy",
        "proxy_feature": "p.npy",
        "start_clip": 0,
        "end_clip": 1,
    }
    write_jsonl(queries, [query, query])
    with pytest.raises(ValueError, match="duplicate query_id"):
        MTVRDataset(queries, videos)
