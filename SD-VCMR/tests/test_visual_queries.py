import json

import numpy as np
import pytest

from scripts.build_visual_queries import annotations


@pytest.mark.parametrize(
    "size,reduced_size",
    [((640, 360), (128, 72)), ((17, 13), (3, 3)), ((2, 1), (1, 1))],
)
def test_bicubic_downsample_and_restore(size, reduced_size):
    image_module = pytest.importorskip("PIL.Image")
    from scripts.visual_features import low_resolution

    width, height = size
    rng = np.random.default_rng(192)
    rgba = rng.integers(0, 256, (height, width, 4), dtype=np.uint8)
    image = image_module.fromarray(rgba)
    reference = image.convert("RGB").resize(reduced_size, image_module.Resampling.BICUBIC)
    expected = reference.resize(size, image_module.Resampling.BICUBIC)

    actual = low_resolution(image)
    assert actual.mode == "RGB"
    assert actual.size == size
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_tvr_jsonl_preserves_annotation_ids_and_seconds(tmp_path):
    rows = [
        {"vid_name": "v1", "desc_id": 7, "desc": "a", "ts": [0, 0.93], "duration": 4.0},
        {"vid_name": "v1", "desc_id": 8, "desc": "b", "ts": [1.5, 3.7], "duration": 4.0},
    ]
    path = tmp_path / "split.jsonl"
    path.write_text("\n\n".join(map(json.dumps, rows)) + "\n", encoding="utf-8")
    assert list(annotations(path, "tvr")) == rows


def test_activitynet_preserves_caption_interval_pairs_and_order(tmp_path):
    source = {"v_abc": {"sentences": ["first", "second"], "timestamps": [[0, 1.25], [3, 6]],
                        "duration": 6}}
    path = tmp_path / "split.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    assert list(annotations(path, "activitynet")) == [
        {"vid_name": "v_abc", "desc_id": "v_abc_0", "desc": "first", "ts": [0, 1.25], "duration": 6},
        {"vid_name": "v_abc", "desc_id": "v_abc_1", "desc": "second", "ts": [3, 6], "duration": 6},
    ]


def test_activitynet_mismatched_number_of_captions_and_spans_fails(tmp_path):
    source = {"v_abc": {"sentences": ["first", "second"], "timestamps": [[0, 1]], "duration": 6}}
    path = tmp_path / "split.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    with pytest.raises(ValueError):
        list(annotations(path, "activitynet"))
