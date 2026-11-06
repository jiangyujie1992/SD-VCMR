from __future__ import annotations

import torch

from mtvr.inference import top_spans
from mtvr.metrics import aggregate_run_metrics, evaluate_predictions, temporal_iou


def test_top_spans_respects_order_and_length() -> None:
    start = torch.tensor([0.05, 0.8, 0.1, 0.05])
    end = torch.tensor([0.05, 0.1, 0.8, 0.05])
    spans = top_spans(start, end, count=3, max_span_length=2)
    assert spans[0][:2] == (1, 2)
    assert all(end_index >= start_index for start_index, end_index, _ in spans)
    assert all(end_index - start_index + 1 <= 2 for start_index, end_index, _ in spans)


def test_temporal_iou_uses_inclusive_clip_indices() -> None:
    assert temporal_iou((1, 2), (1, 2)) == 1.0
    assert temporal_iou((1, 2), (2, 3)) == 1.0 / 3.0


def test_evaluation() -> None:
    predictions = [
        {
            "query_id": "q0",
            "ground_truth": {"video_id": "v0", "start_clip": 2, "end_clip": 4},
            "video_ranking": [
                {"video_id": "v0", "score": 1.0},
                {"video_id": "v1", "score": 0.5},
            ],
            "moments": [
                {
                    "video_id": "v0",
                    "start_clip": 2,
                    "end_clip": 4,
                    "score": 1.0,
                }
            ],
            "svmr_moments": [
                {
                    "video_id": "v0",
                    "start_clip": 2,
                    "end_clip": 4,
                    "score": 1.0,
                }
            ],
        }
    ]
    result = evaluate_predictions(predictions, ks=(1,), ious=(0.5, 0.7))
    assert result["queries"] == 1
    assert result["metrics_percent"]["VR_R@1"] == 100.0
    assert result["metrics_percent"]["VCMR_R@1_IoU0.5"] == 100.0
    assert result["metrics_percent"]["SVMR_R@1_IoU0.7"] == 100.0


def test_seed_aggregation_reports_sample_standard_deviation() -> None:
    first = {
        "queries": 2,
        "protocol": {"ks": [1], "ious": [0.5]},
        "metrics_percent": {"VR_R@1": 40.0},
    }
    second = {
        "queries": 2,
        "protocol": {"ks": [1], "ious": [0.5]},
        "metrics_percent": {"VR_R@1": 60.0},
    }
    aggregate = aggregate_run_metrics([first, second])
    assert aggregate["metrics_percent"]["VR_R@1"]["mean"] == 50.0
    assert aggregate["metrics_percent"]["VR_R@1"]["std"] == 14.142136
