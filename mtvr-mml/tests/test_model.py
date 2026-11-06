from __future__ import annotations

import torch

from mtvr.config import MMLConfig
from mtvr.losses import compute_mml_loss, multi_positive_info_nce
from mtvr.model import MML, retain_top_candidates, sample_candidate_indices


def make_config() -> MMLConfig:
    return MMLConfig(
        video_dim=12,
        subtitle_dim=10,
        query_dim=10,
        proxy_dim=8,
        hidden_dim=16,
        num_heads=4,
        dropout=0.0,
        alpha=0.3,
        tau=0.07,
        lambda_svmr=2.0,
        beta=1.0,
        boundary_kernel=3,
    )


def test_forward_loss_and_backward() -> None:
    torch.manual_seed(3)
    model = MML(make_config())
    batch_size, clips = 3, 6
    mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, True, True, True, False, False],
            [True, True, True, True, True, False],
        ]
    )
    batch = {
        "query_features": torch.randn(batch_size, 10),
        "proxy_features": torch.randn(batch_size, 3, 8),
        "video_features": torch.randn(batch_size, clips, 12),
        "subtitle_features": torch.randn(batch_size, clips, 10),
        "clip_mask": mask,
        "start_clip": torch.tensor([1, 0, 2]),
        "end_clip": torch.tensor([3, 2, 4]),
        "query_video_ids": ["v0", "v1", "v0"],
        "video_ids": ["v0", "v1", "v0"],
    }
    output = model(batch)
    assert output["retrieval_scores"].shape == (batch_size, batch_size)
    assert output["start_logits"].shape == (batch_size, clips)
    assert output["end_logits"].shape == (batch_size, clips)
    assert torch.all(output["start_logits"][~mask] < -1000)
    losses = compute_mml_loss(
        output,
        batch,
        tau=model.config.tau,
        lambda_svmr=model.config.lambda_svmr,
    )
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_multi_positive_loss_accepts_duplicate_video_ids() -> None:
    scores = torch.tensor([[2.0, 0.0, 1.5], [0.0, 2.0, 0.1], [1.5, 0.0, 2.0]])
    loss = multi_positive_info_nce(
        scores,
        ["a", "b", "a"],
        ["a", "b", "a"],
        tau=0.2,
    )
    assert torch.isfinite(loss)
    assert float(loss) < 0.1


def test_candidate_pruning_keeps_positive() -> None:
    scores = torch.tensor([[10.0, 9.0, 8.0, 1.0], [4.0, 3.0, 2.0, 1.0]])
    selected = retain_top_candidates(
        scores, keep=2, positive_indices=torch.tensor([3, 0])
    )
    assert 3 in selected[0].tolist()
    assert 0 in selected[1].tolist()
    assert selected.shape == (2, 2)


def test_sample_and_pruned_forward_are_deterministic_and_keep_positives() -> None:
    first = sample_candidate_indices(
        candidate_count=6,
        sample_size=4,
        positive_indices=torch.tensor([0, 1, 2]),
        generator=torch.Generator().manual_seed(19),
    )
    second = sample_candidate_indices(
        candidate_count=6,
        sample_size=4,
        positive_indices=torch.tensor([0, 1, 2]),
        generator=torch.Generator().manual_seed(19),
    )
    assert torch.equal(first, second)
    assert all(index in first[row].tolist() for row, index in enumerate((0, 1, 2)))

    torch.manual_seed(5)
    model = MML(make_config())
    mask = torch.ones(4, 5, dtype=torch.bool)
    batch = {
        "query_features": torch.randn(4, 10),
        "proxy_features": torch.randn(4, 3, 8),
        "video_features": torch.randn(4, 5, 12),
        "subtitle_features": torch.randn(4, 5, 10),
        "clip_mask": mask,
        "start_clip": torch.tensor([0, 1, 2, 1]),
        "end_clip": torch.tensor([2, 3, 4, 2]),
        "query_video_ids": ["v0", "v1", "v2", "v3"],
        "video_ids": ["v0", "v1", "v2", "v3"],
    }
    output = model.forward_pruned(
        batch,
        sample_size=4,
        keep=2,
        generator=torch.Generator().manual_seed(23),
    )
    assert output["selected_candidate_indices"].shape == (4, 2)
    positives = torch.arange(4)[:, None]
    assert (output["selected_candidate_indices"] == positives).any(dim=1).all()
    assert torch.isfinite(output["retrieval_scores"]).sum(dim=1).tolist() == [2] * 4
    losses = compute_mml_loss(
        output,
        batch,
        tau=model.config.tau,
        lambda_svmr=model.config.lambda_svmr,
    )
    assert torch.isfinite(losses["loss"])
