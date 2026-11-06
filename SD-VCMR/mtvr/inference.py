from __future__ import annotations

import math
from typing import Any

import torch
from torch.nn import functional as F

from .data import VideoStore, pad_video_batch
from .model import MML


def top_spans(
    start_probabilities: torch.Tensor,
    end_probabilities: torch.Tensor,
    *,
    count: int,
    max_span_length: int | None = None,
) -> list[tuple[int, int, float]]:
    if start_probabilities.ndim != 1 or end_probabilities.ndim != 1:
        raise ValueError("boundary probabilities must be one-dimensional")
    if start_probabilities.shape != end_probabilities.shape:
        raise ValueError("start and end probabilities must share a shape")
    if count <= 0:
        return []
    clip_count = start_probabilities.numel()
    starts = torch.arange(clip_count, device=start_probabilities.device)[:, None]
    ends = torch.arange(clip_count, device=start_probabilities.device)[None, :]
    valid = ends >= starts
    if max_span_length is not None:
        if max_span_length <= 0:
            raise ValueError("max_span_length must be positive")
        valid &= (ends - starts + 1) <= max_span_length
    scores = start_probabilities[:, None] * end_probabilities[None, :]
    scores = scores.masked_fill(~valid, -1.0)
    keep = min(count, int(valid.sum().item()))
    values, indices = scores.flatten().topk(keep)
    output: list[tuple[int, int, float]] = []
    for value, index in zip(values.tolist(), indices.tolist()):
        start = index // clip_count
        end = index % clip_count
        output.append((int(start), int(end), float(value)))
    return output


def _move_video_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@torch.inference_mode()
def score_corpus(
    model: MML,
    query_features: torch.Tensor,
    proxy_features: torch.Tensor,
    videos: VideoStore,
    *,
    batch_size: int,
    device: torch.device,
) -> list[tuple[str, float]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    model.eval()
    query_features = query_features.unsqueeze(0).to(device)
    proxy_features = proxy_features.unsqueeze(0).to(device)
    scored: list[tuple[str, float]] = []
    video_ids = videos.video_ids
    for offset in range(0, len(video_ids), batch_size):
        ids = video_ids[offset : offset + batch_size]
        records = [videos.load(video_id) for video_id in ids]
        batch = _move_video_batch(pad_video_batch(records), device)
        scores = model.score_videos(
            query_features,
            proxy_features,
            batch["video_features"],
            batch["subtitle_features"],
            batch["clip_mask"],
        )[0]
        scored.extend((video_id, float(score)) for video_id, score in zip(ids, scores))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


@torch.inference_mode()
def localize_videos(
    model: MML,
    query_features: torch.Tensor,
    proxy_features: torch.Tensor,
    video_records: list[dict[str, Any]],
    *,
    spans_per_video: int,
    max_span_length: int | None,
    device: torch.device,
) -> dict[str, list[tuple[int, int, float]]]:
    if not video_records:
        return {}
    model.eval()
    batch = _move_video_batch(pad_video_batch(video_records), device)
    count = len(video_records)
    queries = query_features.unsqueeze(0).expand(count, -1).to(device)
    proxies = proxy_features.unsqueeze(0).expand(count, -1, -1).to(device)
    output = model.localize_pairs(
        queries,
        proxies,
        batch["video_features"],
        batch["subtitle_features"],
        batch["clip_mask"],
    )
    result: dict[str, list[tuple[int, int, float]]] = {}
    for index, video_id in enumerate(batch["video_ids"]):
        valid_clips = int(batch["clip_mask"][index].sum().item())
        start = F.softmax(output["start_logits"][index, :valid_clips], dim=0)
        end = F.softmax(output["end_logits"][index, :valid_clips], dim=0)
        result[video_id] = top_spans(
            start,
            end,
            count=spans_per_video,
            max_span_length=max_span_length,
        )
    return result


@torch.inference_mode()
def predict_query(
    model: MML,
    query: dict[str, Any],
    videos: VideoStore,
    *,
    video_batch_size: int,
    top_videos: int,
    spans_per_video: int,
    svmr_spans: int,
    max_span_length: int | None,
    beta: float,
    device: torch.device,
) -> dict[str, Any]:
    if top_videos <= 0 or spans_per_video <= 0 or svmr_spans <= 0:
        raise ValueError("top_videos and span counts must be positive")
    if video_batch_size <= 0:
        raise ValueError("video_batch_size must be positive")
    ranking = score_corpus(
        model,
        query["query_features"],
        query["proxy_features"],
        videos,
        batch_size=video_batch_size,
        device=device,
    )
    ranking = ranking[: min(top_videos, len(ranking))]
    candidates = [videos.load(video_id) for video_id, _ in ranking]
    localized: dict[str, list[tuple[int, int, float]]] = {}
    for offset in range(0, len(candidates), video_batch_size):
        localized.update(
            localize_videos(
                model,
                query["query_features"],
                query["proxy_features"],
                candidates[offset : offset + video_batch_size],
                spans_per_video=spans_per_video,
                max_span_length=max_span_length,
                device=device,
            )
        )
    video_score = dict(ranking)
    moments: list[dict[str, Any]] = []
    for video_id, spans in localized.items():
        retrieval_factor = math.exp(beta * video_score[video_id])
        for start, end, span_score in spans:
            moments.append(
                {
                    "video_id": video_id,
                    "start_clip": start,
                    "end_clip": end,
                    "score": span_score * retrieval_factor,
                }
            )
    moments.sort(key=lambda item: item["score"], reverse=True)
    ground_truth: dict[str, Any] | None = None
    svmr_moments: list[dict[str, Any]] = []
    if "start_clip" in query and "end_clip" in query:
        ground_truth = {
            "video_id": query["video_id"],
            "start_clip": int(query["start_clip"]),
            "end_clip": int(query["end_clip"]),
        }
        gt_record = videos.load(query["video_id"])
        gt_spans = localize_videos(
            model,
            query["query_features"],
            query["proxy_features"],
            [gt_record],
            spans_per_video=svmr_spans,
            max_span_length=max_span_length,
            device=device,
        )[query["video_id"]]
        svmr_moments = [
            {
                "video_id": query["video_id"],
                "start_clip": start,
                "end_clip": end,
                "score": score,
            }
            for start, end, score in gt_spans
        ]
    return {
        "query_id": query["query_id"],
        "video_ranking": [
            {"video_id": video_id, "score": score} for video_id, score in ranking
        ],
        "moments": moments,
        "svmr_moments": svmr_moments,
        "ground_truth": ground_truth,
    }
