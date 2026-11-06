from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import MMLConfig


class FeatureProjector(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)
        self.normalization = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.normalization(self.projection(features)))


class MML(nn.Module):
    """Feature-level implementation of Multi-modal Moment Localization.

    The method mapping is documented in ``docs/IMPLEMENTATION.md``. In brief,
    the four projectors implement the frozen-feature projections in Eqs. (1)-
    (4), bidirectional attention implements Eqs. (5)-(6), clip matching and
    boundary heads implement Eqs. (7)-(12), and retrieval implements Eq. (13).
    """

    def __init__(self, config: MMLConfig) -> None:
        super().__init__()
        self.config = config
        self.video_projector = FeatureProjector(
            config.video_dim, config.hidden_dim, config.dropout
        )
        self.subtitle_projector = FeatureProjector(
            config.subtitle_dim, config.hidden_dim, config.dropout
        )
        self.query_projector = FeatureProjector(
            config.query_dim, config.hidden_dim, config.dropout
        )
        self.proxy_projector = FeatureProjector(
            config.proxy_dim, config.hidden_dim, config.dropout
        )
        self.video_to_subtitle = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.subtitle_to_video = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self.video_context_norm = nn.LayerNorm(config.hidden_dim)
        self.subtitle_context_norm = nn.LayerNorm(config.hidden_dim)
        self.context_dropout = nn.Dropout(config.dropout)
        padding = config.boundary_kernel // 2
        self.start_head = nn.Conv1d(
            1, 1, config.boundary_kernel, padding=padding, bias=False
        )
        self.end_head = nn.Conv1d(
            1, 1, config.boundary_kernel, padding=padding, bias=False
        )

    def encode_query(
        self, query_features: torch.Tensor, proxy_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_features.ndim != 2:
            raise ValueError("query_features must have shape [B, query_dim]")
        if proxy_features.ndim != 3:
            raise ValueError("proxy_features must have shape [B, styles, proxy_dim]")
        if query_features.shape[0] != proxy_features.shape[0]:
            raise ValueError("query and proxy batch sizes must match")
        if query_features.shape[-1] != self.config.query_dim:
            raise ValueError("query feature width does not match model configuration")
        if proxy_features.shape[-1] != self.config.proxy_dim:
            raise ValueError("proxy feature width does not match model configuration")
        semantic_query = self.query_projector(query_features)
        proxy_tokens = self.proxy_projector(proxy_features)
        # Equation (2): average pooling produces a style-agnostic spatial query.
        spatial_query = proxy_tokens.mean(dim=1)
        return semantic_query, spatial_query

    def encode_video(
        self, video_features: torch.Tensor, subtitle_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if video_features.ndim != 3 or subtitle_features.ndim != 3:
            raise ValueError("video and subtitle features must have shape [B, T, D]")
        if video_features.shape[:2] != subtitle_features.shape[:2]:
            raise ValueError("video and subtitle batch/clip dimensions must match")
        if video_features.shape[-1] != self.config.video_dim:
            raise ValueError("video feature width does not match model configuration")
        if subtitle_features.shape[-1] != self.config.subtitle_dim:
            raise ValueError("subtitle feature width does not match model configuration")
        return (
            self.video_projector(video_features),
            self.subtitle_projector(subtitle_features),
        )

    @staticmethod
    def _validate_mask(clip_mask: torch.Tensor, shape: tuple[int, int]) -> None:
        if clip_mask.dtype != torch.bool or tuple(clip_mask.shape) != shape:
            raise ValueError("clip_mask must be boolean with shape [B, T]")
        if not clip_mask.any(dim=1).all():
            raise ValueError("each video must contain at least one valid clip")

    @staticmethod
    def _masked_clip_cosine(
        clips: torch.Tensor, queries: torch.Tensor, clip_mask: torch.Tensor
    ) -> torch.Tensor:
        scores = F.cosine_similarity(clips, queries[:, None, :], dim=-1)
        return scores.masked_fill(~clip_mask, -1.0e4)

    @staticmethod
    def _score_matrix(
        query_embeddings: torch.Tensor,
        clip_embeddings: torch.Tensor,
        clip_mask: torch.Tensor,
    ) -> torch.Tensor:
        queries = F.normalize(query_embeddings, dim=-1)
        clips = F.normalize(clip_embeddings, dim=-1)
        scores = torch.einsum("qh,vth->qvt", queries, clips)
        scores = scores.masked_fill(~clip_mask[None, :, :], -1.0e4)
        return scores.max(dim=-1).values

    def contextualize_video(
        self,
        video_embeddings: torch.Tensor,
        subtitle_embeddings: torch.Tensor,
        clip_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the bidirectional video/subtitle interaction in Eqs. (5)-(6)."""

        if video_embeddings.shape != subtitle_embeddings.shape:
            raise ValueError("projected video and subtitle tensors must share a shape")
        self._validate_mask(clip_mask, tuple(video_embeddings.shape[:2]))
        padding_mask = ~clip_mask
        video_context, _ = self.video_to_subtitle(
            video_embeddings,
            subtitle_embeddings,
            subtitle_embeddings,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        subtitle_context, _ = self.subtitle_to_video(
            subtitle_embeddings,
            video_embeddings,
            video_embeddings,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        video_context = self.video_context_norm(
            video_embeddings + self.context_dropout(video_context)
        )
        subtitle_context = self.subtitle_context_norm(
            subtitle_embeddings + self.context_dropout(subtitle_context)
        )
        return video_context, subtitle_context

    def score_videos_from_embeddings(
        self,
        semantic_queries: torch.Tensor,
        spatial_queries: torch.Tensor,
        video_context: torch.Tensor,
        subtitle_context: torch.Tensor,
        clip_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the max-pooled two-branch video score from Eq. (13)."""

        spatial = self._score_matrix(spatial_queries, video_context, clip_mask)
        semantic = self._score_matrix(semantic_queries, subtitle_context, clip_mask)
        return 0.5 * (spatial + semantic)

    def score_videos(
        self,
        query_features: torch.Tensor,
        proxy_features: torch.Tensor,
        video_features: torch.Tensor,
        subtitle_features: torch.Tensor,
        clip_mask: torch.Tensor,
    ) -> torch.Tensor:
        semantic_queries, spatial_queries = self.encode_query(
            query_features, proxy_features
        )
        video_embeddings, subtitle_embeddings = self.encode_video(
            video_features, subtitle_features
        )
        video_context, subtitle_context = self.contextualize_video(
            video_embeddings, subtitle_embeddings, clip_mask
        )
        return self.score_videos_from_embeddings(
            semantic_queries,
            spatial_queries,
            video_context,
            subtitle_context,
            clip_mask,
        )

    def localize_contextualized_pairs(
        self,
        semantic_queries: torch.Tensor,
        spatial_queries: torch.Tensor,
        video_context: torch.Tensor,
        subtitle_context: torch.Tensor,
        clip_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        batch_size = semantic_queries.shape[0]
        if not (
            spatial_queries.shape[0]
            == video_context.shape[0]
            == subtitle_context.shape[0]
            == clip_mask.shape[0]
            == batch_size
        ):
            raise ValueError("paired localization inputs must share a batch size")
        spatial_scores = self._masked_clip_cosine(
            video_context, spatial_queries, clip_mask
        )
        semantic_scores = self._masked_clip_cosine(
            subtitle_context, semantic_queries, clip_mask
        )
        # Equation (9): alpha is the semantic branch weight.
        clip_scores = (
            (1.0 - self.config.alpha) * spatial_scores
            + self.config.alpha * semantic_scores
        )
        head_input = clip_scores.masked_fill(~clip_mask, 0.0).unsqueeze(1)
        start_logits = self.start_head(head_input).squeeze(1)
        end_logits = self.end_head(head_input).squeeze(1)
        start_logits = start_logits.masked_fill(~clip_mask, -1.0e4)
        end_logits = end_logits.masked_fill(~clip_mask, -1.0e4)
        return {
            "spatial_scores": spatial_scores,
            "semantic_scores": semantic_scores,
            "clip_scores": clip_scores,
            "start_logits": start_logits,
            "end_logits": end_logits,
        }

    def localize_pairs_from_embeddings(
        self,
        semantic_queries: torch.Tensor,
        spatial_queries: torch.Tensor,
        video_embeddings: torch.Tensor,
        subtitle_embeddings: torch.Tensor,
        clip_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        video_context, subtitle_context = self.contextualize_video(
            video_embeddings, subtitle_embeddings, clip_mask
        )
        return self.localize_contextualized_pairs(
            semantic_queries,
            spatial_queries,
            video_context,
            subtitle_context,
            clip_mask,
        )

    def localize_pairs(
        self,
        query_features: torch.Tensor,
        proxy_features: torch.Tensor,
        video_features: torch.Tensor,
        subtitle_features: torch.Tensor,
        clip_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        semantic_queries, spatial_queries = self.encode_query(
            query_features, proxy_features
        )
        video_embeddings, subtitle_embeddings = self.encode_video(
            video_features, subtitle_features
        )
        return self.localize_pairs_from_embeddings(
            semantic_queries,
            spatial_queries,
            video_embeddings,
            subtitle_embeddings,
            clip_mask,
        )

    def cheap_spatial_scores(
        self,
        proxy_features: torch.Tensor,
        video_features: torch.Tensor,
        clip_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Training-only pre-filter score used before cross-attention."""

        spatial_queries = self.proxy_projector(proxy_features).mean(dim=1)
        video_embeddings = self.video_projector(video_features)
        return self._score_matrix(spatial_queries, video_embeddings, clip_mask)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Unpruned in-batch forward pass, primarily useful for diagnostics."""

        semantic_queries, spatial_queries = self.encode_query(
            batch["query_features"], batch["proxy_features"]
        )
        video_embeddings, subtitle_embeddings = self.encode_video(
            batch["video_features"], batch["subtitle_features"]
        )
        video_context, subtitle_context = self.contextualize_video(
            video_embeddings, subtitle_embeddings, batch["clip_mask"]
        )
        localization = self.localize_contextualized_pairs(
            semantic_queries,
            spatial_queries,
            video_context,
            subtitle_context,
            batch["clip_mask"],
        )
        retrieval_scores = self.score_videos_from_embeddings(
            semantic_queries,
            spatial_queries,
            video_context,
            subtitle_context,
            batch["clip_mask"],
        )
        return {**localization, "retrieval_scores": retrieval_scores}

    def forward_pruned(
        self,
        batch: dict[str, Any],
        *,
        sample_size: int,
        keep: int,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        """Paper training path: sample candidates, spatially prune, then attend.

        The candidate pool is the in-batch video set from Eq. (14). For every
        query, at most ``sample_size`` candidates are sampled (always including
        its paired video); the cheap pooled spatial score retains ``keep`` hard
        candidates before video/subtitle cross-attention. Unselected entries in
        ``retrieval_scores`` are ``-inf`` and therefore excluded from InfoNCE.
        """

        semantic_queries, spatial_queries = self.encode_query(
            batch["query_features"], batch["proxy_features"]
        )
        video_embeddings, subtitle_embeddings = self.encode_video(
            batch["video_features"], batch["subtitle_features"]
        )
        clip_mask = batch["clip_mask"]
        batch_size = semantic_queries.shape[0]
        if video_embeddings.shape[0] != batch_size:
            raise ValueError("training requires one paired candidate video per query")
        effective_sample = min(sample_size, batch_size)
        effective_keep = min(keep, effective_sample)
        if effective_sample <= 0 or effective_keep <= 0:
            raise ValueError("sample_size and keep must be positive")

        positive_indices = torch.arange(batch_size, dtype=torch.long)
        sampled = sample_candidate_indices(
            candidate_count=batch_size,
            sample_size=effective_sample,
            positive_indices=positive_indices,
            generator=generator,
        ).to(video_embeddings.device)
        cheap_scores = self._score_matrix(spatial_queries, video_embeddings, clip_mask)
        sampled_scores = cheap_scores.gather(1, sampled)
        positive_positions = (sampled == positive_indices.to(sampled.device)[:, None]).to(
            torch.int64
        ).argmax(dim=1)
        retained_positions = retain_top_candidates(
            sampled_scores,
            keep=effective_keep,
            positive_indices=positive_positions,
        )
        selected = sampled.gather(1, retained_positions)

        flat_selected = selected.reshape(-1)
        query_count, candidate_keep = selected.shape
        semantic_pairs = semantic_queries[:, None, :].expand(
            -1, candidate_keep, -1
        ).reshape(query_count * candidate_keep, -1)
        spatial_pairs = spatial_queries[:, None, :].expand(
            -1, candidate_keep, -1
        ).reshape(query_count * candidate_keep, -1)
        pair_mask = clip_mask[flat_selected]
        pair_output = self.localize_pairs_from_embeddings(
            semantic_pairs,
            spatial_pairs,
            video_embeddings[flat_selected],
            subtitle_embeddings[flat_selected],
            pair_mask,
        )
        pair_scores = 0.5 * (
            pair_output["spatial_scores"].max(dim=1).values
            + pair_output["semantic_scores"].max(dim=1).values
        )
        selected_scores = pair_scores.reshape(query_count, candidate_keep)
        retrieval_scores = torch.full(
            (batch_size, batch_size),
            -torch.inf,
            dtype=selected_scores.dtype,
            device=selected_scores.device,
        )
        retrieval_scores.scatter_(1, selected, selected_scores)

        positive_columns = (
            selected == torch.arange(batch_size, device=selected.device)[:, None]
        ).to(torch.int64).argmax(dim=1)
        positive_rows = (
            torch.arange(batch_size, device=selected.device) * candidate_keep
            + positive_columns
        )
        return {
            "retrieval_scores": retrieval_scores,
            "start_logits": pair_output["start_logits"][positive_rows],
            "end_logits": pair_output["end_logits"][positive_rows],
            "spatial_scores": pair_output["spatial_scores"][positive_rows],
            "semantic_scores": pair_output["semantic_scores"][positive_rows],
            "clip_scores": pair_output["clip_scores"][positive_rows],
            "sampled_candidate_indices": sampled,
            "selected_candidate_indices": selected,
            "cheap_retrieval_scores": cheap_scores,
        }


def sample_candidate_indices(
    *,
    candidate_count: int,
    sample_size: int,
    positive_indices: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    """Sample unique candidate columns per query while preserving each positive."""

    if candidate_count <= 0 or not 0 < sample_size <= candidate_count:
        raise ValueError("sample_size must be between 1 and candidate_count")
    if positive_indices.shape != (positive_indices.numel(),):
        raise ValueError("positive_indices must be one-dimensional")
    if ((positive_indices < 0) | (positive_indices >= candidate_count)).any():
        raise ValueError("positive candidate index is out of range")
    rows = []
    for positive in positive_indices.cpu().tolist():
        if sample_size == candidate_count:
            sampled = torch.arange(candidate_count, dtype=torch.long)
        else:
            sampled = torch.randperm(candidate_count, generator=generator)[:sample_size]
            if not (sampled == positive).any():
                sampled[-1] = positive
        rows.append(sampled)
    return torch.stack(rows, dim=0)


def retain_top_candidates(
    scores: torch.Tensor,
    keep: int,
    positive_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    if scores.ndim != 2:
        raise ValueError("scores must have shape [queries, candidates]")
    candidate_count = scores.shape[1]
    if not 0 < keep <= candidate_count:
        raise ValueError("keep must be between 1 and the candidate count")
    selected = scores.topk(keep, dim=1).indices
    if positive_indices is None:
        return selected
    if positive_indices.shape != (scores.shape[0],):
        raise ValueError("positive_indices must have shape [queries]")
    if ((positive_indices < 0) | (positive_indices >= candidate_count)).any():
        raise ValueError("positive candidate index is out of range")
    selected = selected.clone()
    present = (selected == positive_indices[:, None]).any(dim=1)
    selected[~present, -1] = positive_indices[~present]
    return selected
