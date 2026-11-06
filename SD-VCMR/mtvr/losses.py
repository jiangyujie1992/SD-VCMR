from __future__ import annotations

from typing import Any, Sequence

import torch
from torch.nn import functional as F


def multi_positive_info_nce(
    scores: torch.Tensor,
    query_video_ids: Sequence[str],
    candidate_video_ids: Sequence[str],
    tau: float,
) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must have shape [queries, candidates]")
    if scores.shape != (len(query_video_ids), len(candidate_video_ids)):
        raise ValueError("score shape does not match the supplied ID lists")
    if tau <= 0.0:
        raise ValueError("tau must be positive")
    positive = torch.tensor(
        [
            [query_id == candidate_id for candidate_id in candidate_video_ids]
            for query_id in query_video_ids
        ],
        dtype=torch.bool,
        device=scores.device,
    )
    if not positive.any(dim=1).all():
        raise ValueError("every query requires at least one positive video")
    logits = scores / tau
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise ValueError("retrieval scores contain NaN or positive infinity")
    if not torch.isfinite(logits.masked_fill(~positive, -torch.inf)).any(dim=1).all():
        raise ValueError("every query requires a finite positive score")
    if not torch.isfinite(logits).any(dim=1).all():
        raise ValueError("every query requires at least one retained candidate")
    numerator = torch.logsumexp(logits.masked_fill(~positive, -torch.inf), dim=1)
    denominator = torch.logsumexp(logits, dim=1)
    return (denominator - numerator).mean()


def boundary_loss(
    start_logits: torch.Tensor,
    end_logits: torch.Tensor,
    start_clip: torch.Tensor,
    end_clip: torch.Tensor,
    clip_mask: torch.Tensor,
) -> torch.Tensor:
    if start_logits.shape != end_logits.shape or start_logits.shape != clip_mask.shape:
        raise ValueError("boundary logits and clip_mask must share shape [B, T]")
    if start_clip.shape != end_clip.shape or start_clip.shape != (start_logits.shape[0],):
        raise ValueError("boundary targets must have shape [B]")
    if (end_clip < start_clip).any():
        raise ValueError("end targets must not precede start targets")
    row = torch.arange(start_logits.shape[0], device=start_logits.device)
    if (
        (start_clip < 0).any()
        or (end_clip >= start_logits.shape[1]).any()
        or not clip_mask[row, start_clip].all()
        or not clip_mask[row, end_clip].all()
    ):
        raise ValueError("boundary target points to a padded or invalid clip")
    return F.cross_entropy(start_logits, start_clip) + F.cross_entropy(
        end_logits, end_clip
    )


def compute_mml_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    *,
    tau: float,
    lambda_svmr: float,
) -> dict[str, torch.Tensor]:
    retrieval = multi_positive_info_nce(
        outputs["retrieval_scores"],
        batch["query_video_ids"],
        batch["video_ids"],
        tau,
    )
    localization = boundary_loss(
        outputs["start_logits"],
        outputs["end_logits"],
        batch["start_clip"],
        batch["end_clip"],
        batch["clip_mask"],
    )
    total = retrieval + lambda_svmr * localization
    return {
        "loss": total,
        "loss_vr": retrieval.detach(),
        "loss_svmr": localization.detach(),
    }
