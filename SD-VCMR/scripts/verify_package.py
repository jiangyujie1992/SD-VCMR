from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtvr.config import load_experiment_config
from mtvr.runtime import sha256_file


EXCLUDED_DIRECTORIES = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "checkpoints",
    "outputs",
    "runs",
    "tmp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify or regenerate the anonymous artifact inventory"
    )
    parser.add_argument(
        "--write-checksums",
        action="store_true",
        help="regenerate CHECKSUMS.sha256 after an intentional release change",
    )
    return parser.parse_args()


def released_files() -> set[str]:
    output = set()
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.name == "CHECKSUMS.sha256":
            continue
        relative = path.relative_to(ROOT)
        if any(
            part in EXCLUDED_DIRECTORIES or part.endswith(".egg-info")
            for part in relative.parts
        ):
            continue
        if path.suffix == ".pyc":
            continue
        output.add(relative.as_posix())
    return output


def write_checksums() -> int:
    paths = sorted(released_files())
    text = "".join(f"{sha256_file(ROOT / path)}  {path}\n" for path in paths)
    destination = ROOT / "CHECKSUMS.sha256"
    temporary = destination.with_suffix(".sha256.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(destination)
    return len(paths)


def verify_checksums() -> int:
    checksum_path = ROOT / "CHECKSUMS.sha256"
    if not checksum_path.is_file():
        raise SystemExit("CHECKSUMS.sha256 is missing")
    expected = {}
    for line_number, line in enumerate(
        checksum_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            checksum, relative = line.split("  ", 1)
        except ValueError as exc:
            raise SystemExit(f"invalid checksum line {line_number}") from exc
        if len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum):
            raise SystemExit(f"invalid SHA-256 digest on line {line_number}")
        if relative in expected:
            raise SystemExit(f"duplicate checksum entry: {relative}")
        expected[relative] = checksum
    actual = released_files()
    if actual != set(expected):
        raise SystemExit(
            f"inventory mismatch: unexpected={sorted(actual - set(expected))}, "
            f"missing={sorted(set(expected) - actual)}"
        )
    failures = [
        relative
        for relative, checksum in expected.items()
        if sha256_file(ROOT / relative) != checksum
    ]
    if failures:
        raise SystemExit(f"checksum failures: {failures}")
    return len(expected)


def verify_model() -> dict[str, object]:
    import torch

    from mtvr.config import MMLConfig
    from mtvr.losses import compute_mml_loss
    from mtvr.model import MML

    torch.manual_seed(11)
    config = MMLConfig(
        video_dim=8,
        subtitle_dim=8,
        query_dim=8,
        proxy_dim=8,
        hidden_dim=16,
        num_heads=4,
        dropout=0.0,
        alpha=0.3,
        tau=0.07,
        lambda_svmr=25.0,
        beta=1.0,
        boundary_kernel=3,
    )
    model = MML(config)
    mask = torch.tensor(
        [
            [True] * 5,
            [True] * 4 + [False],
            [True] * 5,
            [True] * 3 + [False] * 2,
        ]
    )
    batch = {
        "query_features": torch.randn(4, 8),
        "proxy_features": torch.randn(4, 3, 8),
        "video_features": torch.randn(4, 5, 8),
        "subtitle_features": torch.randn(4, 5, 8),
        "clip_mask": mask,
        "start_clip": torch.tensor([1, 0, 2, 0]),
        "end_clip": torch.tensor([3, 2, 4, 1]),
        "query_video_ids": ["v0", "v1", "v2", "v3"],
        "video_ids": ["v0", "v1", "v2", "v3"],
    }
    output = model.forward_pruned(
        batch,
        sample_size=4,
        keep=2,
        generator=torch.Generator().manual_seed(12),
    )
    losses = compute_mml_loss(
        output,
        batch,
        tau=config.tau,
        lambda_svmr=config.lambda_svmr,
    )
    losses["loss"].backward()
    if not math.isfinite(float(losses["loss"].detach())):
        raise SystemExit("non-finite model loss")
    selected = output["selected_candidate_indices"]
    positives = torch.arange(4)[:, None]
    if not (selected == positives).any(dim=1).all():
        raise SystemExit("candidate pruning did not preserve every positive")
    retained = torch.isfinite(output["retrieval_scores"]).sum(dim=1)
    if retained.tolist() != [2, 2, 2, 2]:
        raise SystemExit("candidate pruning retained an unexpected number of videos")
    return {
        "boundary_shape": list(output["start_logits"].shape),
        "loss_finite": True,
        "backward": True,
        "candidate_sample_size": 4,
        "candidate_keep": 2,
        "positive_retained": True,
    }


def main() -> None:
    args = parse_args()
    if args.write_checksums:
        count = write_checksums()
        print(json.dumps({"status": "written", "released_files": count}, indent=2))
        return
    file_count = verify_checksums()
    config = load_experiment_config(ROOT / "configs" / "tvr_mml.json")
    model_status = verify_model()
    print(
        json.dumps(
            {
                "status": "pass",
                "released_files": file_count,
                "config_schema_version": config.schema_version,
                "model": model_status,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
