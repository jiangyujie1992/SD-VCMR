from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any, Iterable


def temporal_iou(
    prediction: tuple[int, int], ground_truth: tuple[int, int]
) -> float:
    """Temporal IoU for inclusive, zero-based clip boundaries."""

    pred_start, pred_end = prediction
    gt_start, gt_end = ground_truth
    if min(pred_start, pred_end, gt_start, gt_end) < 0:
        raise ValueError("temporal clip indices must be non-negative")
    if pred_end < pred_start or gt_end < gt_start:
        raise ValueError("temporal spans must satisfy start <= end")
    pred_end_exclusive = pred_end + 1
    gt_end_exclusive = gt_end + 1
    intersection = max(
        0, min(pred_end_exclusive, gt_end_exclusive) - max(pred_start, gt_start)
    )
    union = max(pred_end_exclusive, gt_end_exclusive) - min(pred_start, gt_start)
    return float(intersection / union) if union else 0.0


def _validate_ranked_items(
    query_id: str, name: str, items: Any, required: tuple[str, ...]
) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        raise ValueError(f"{query_id}: {name} must be a list")
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{query_id}: {name}[{index}] must be an object")
        missing = sorted(set(required) - set(item))
        if missing:
            raise ValueError(f"{query_id}: {name}[{index}] is missing {missing}")
    return items


def evaluate_predictions(
    predictions: Iterable[dict[str, Any]],
    *,
    ks: tuple[int, ...] = (1, 10, 100),
    ious: tuple[float, ...] = (0.5, 0.7),
) -> dict[str, Any]:
    """Evaluate VR, VCMR, and SVMR Recall@K using the TVR protocol."""

    if not ks or any(k <= 0 for k in ks):
        raise ValueError("ks must contain positive integers")
    if len(set(ks)) != len(ks):
        raise ValueError("ks must be unique")
    if not ious or any(not 0.0 <= threshold <= 1.0 for threshold in ious):
        raise ValueError("IoU thresholds must lie in [0, 1]")
    if len(set(ious)) != len(ious):
        raise ValueError("IoU thresholds must be unique")
    counts: defaultdict[str, int] = defaultdict(int)
    seen: set[str] = set()
    total = 0
    for record in predictions:
        if not isinstance(record, dict):
            raise ValueError("each prediction must be an object")
        query_id = str(record.get("query_id", ""))
        if not query_id:
            raise ValueError("each prediction requires query_id")
        if query_id in seen:
            raise ValueError(f"duplicate prediction for query_id: {query_id}")
        seen.add(query_id)
        ground_truth = record.get("ground_truth")
        if not isinstance(ground_truth, dict):
            raise ValueError(f"{query_id}: ground_truth is required for evaluation")
        required_gt = {"video_id", "start_clip", "end_clip"}
        if required_gt - set(ground_truth):
            raise ValueError(f"{query_id}: incomplete ground_truth")
        total += 1
        gt_video = str(ground_truth["video_id"])
        gt_span = (int(ground_truth["start_clip"]), int(ground_truth["end_clip"]))
        temporal_iou(gt_span, gt_span)  # validates the ground-truth span
        ranking = _validate_ranked_items(
            query_id, "video_ranking", record.get("video_ranking"), ("video_id",)
        )
        moments = _validate_ranked_items(
            query_id,
            "moments",
            record.get("moments"),
            ("video_id", "start_clip", "end_clip"),
        )
        svmr = _validate_ranked_items(
            query_id,
            "svmr_moments",
            record.get("svmr_moments"),
            ("video_id", "start_clip", "end_clip"),
        )
        for k in ks:
            if any(str(item["video_id"]) == gt_video for item in ranking[:k]):
                counts[f"VR_R@{k}"] += 1
            for threshold in ious:
                if any(
                    str(item["video_id"]) == gt_video
                    and temporal_iou(
                        (int(item["start_clip"]), int(item["end_clip"])), gt_span
                    )
                    >= threshold
                    for item in moments[:k]
                ):
                    counts[f"VCMR_R@{k}_IoU{threshold:g}"] += 1
                if any(
                    str(item["video_id"]) == gt_video
                    and temporal_iou(
                        (int(item["start_clip"]), int(item["end_clip"])), gt_span
                    )
                    >= threshold
                    for item in svmr[:k]
                ):
                    counts[f"SVMR_R@{k}_IoU{threshold:g}"] += 1
    if total == 0:
        raise ValueError("prediction file is empty")
    metrics = {
        key: round(100.0 * value / total, 6) for key, value in sorted(counts.items())
    }
    for k in ks:
        metrics.setdefault(f"VR_R@{k}", 0.0)
        for threshold in ious:
            metrics.setdefault(f"VCMR_R@{k}_IoU{threshold:g}", 0.0)
            metrics.setdefault(f"SVMR_R@{k}_IoU{threshold:g}", 0.0)
    return {
        "queries": total,
        "protocol": {
            "clip_boundaries": "inclusive_zero_based",
            "ks": list(ks),
            "ious": list(ious),
        },
        "metrics_percent": dict(sorted(metrics.items())),
    }


def aggregate_run_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Report mean, sample standard deviation, and all per-run values."""

    if not results:
        raise ValueError("at least one run is required")
    metric_keys = set(results[0]["metrics_percent"])
    if any(set(result["metrics_percent"]) != metric_keys for result in results[1:]):
        raise ValueError("all runs must contain the same metric keys")
    aggregate = {}
    for key in sorted(metric_keys):
        values = [float(result["metrics_percent"][key]) for result in results]
        aggregate[key] = {
            "mean": round(statistics.fmean(values), 6),
            "std": round(statistics.stdev(values), 6) if len(values) > 1 else 0.0,
            "values": values,
        }
    return {
        "runs": len(results),
        "queries_per_run": [int(result["queries"]) for result in results],
        "protocol": results[0]["protocol"],
        "metrics_percent": aggregate,
    }
