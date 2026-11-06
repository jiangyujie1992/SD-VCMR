from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtvr.data import write_jsonl
from mtvr.runtime import atomic_write_json, sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create deterministic synthetic data for the MTVR smoke test"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--videos", type=int, default=8)
    parser.add_argument("--queries-per-video", type=int, default=3)
    parser.add_argument(
        "--overwrite", action="store_true", help="allow replacing an existing smoke dataset"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.videos < 2 or args.queries_per_video < 2:
        raise ValueError("smoke data requires at least 2 videos and 2 queries per video")
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"output directory is not empty: {output}; pass --overwrite to replace files"
        )
    feature_root = output / "features"
    for name in ("video", "subtitle", "query", "proxy"):
        (feature_root / name).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    dimension = 16
    video_records = []
    train_queries = []
    val_queries = []
    for video_index in range(args.videos):
        video_id = f"video_{video_index:03d}"
        clip_count = 7 + video_index % 3
        latent = rng.normal(size=dimension).astype(np.float32)
        latent /= max(np.linalg.norm(latent), 1.0e-6)
        start = 1 + video_index % (clip_count - 3)
        end = min(start + 1, clip_count - 1)
        video = rng.normal(scale=0.2, size=(clip_count, dimension)).astype(np.float32)
        subtitle = rng.normal(
            scale=0.2, size=(clip_count, dimension)
        ).astype(np.float32)
        video[start : end + 1] += 2.5 * latent
        subtitle[start : end + 1] += 2.0 * latent
        video_path = feature_root / "video" / f"{video_id}.npy"
        subtitle_path = feature_root / "subtitle" / f"{video_id}.npy"
        np.save(video_path, video)
        np.save(subtitle_path, subtitle)
        video_records.append(
            {
                "video_id": video_id,
                "video_feature": video_path.relative_to(output).as_posix(),
                "subtitle_feature": subtitle_path.relative_to(output).as_posix(),
                "duration": 1.5 * clip_count,
                "clip_duration": 1.5,
            }
        )
        for query_index in range(args.queries_per_video):
            query_id = f"query_{video_index:03d}_{query_index:02d}"
            query = latent + rng.normal(scale=0.08, size=dimension).astype(np.float32)
            proxy = np.stack(
                [
                    latent
                    + rng.normal(scale=0.1 + 0.03 * style, size=dimension).astype(
                        np.float32
                    )
                    for style in range(3)
                ],
                axis=0,
            )
            query_path = feature_root / "query" / f"{query_id}.npy"
            proxy_path = feature_root / "proxy" / f"{query_id}.npy"
            np.save(query_path, query)
            np.save(proxy_path, proxy)
            record = {
                "query_id": query_id,
                "video_id": video_id,
                "query_feature": query_path.relative_to(output).as_posix(),
                "proxy_feature": proxy_path.relative_to(output).as_posix(),
                "start_clip": start,
                "end_clip": end,
                "query_text": f"toy event in {video_id}",
                "enhanced_text": f"toy event with spatial hint in {video_id}",
            }
            if query_index == args.queries_per_video - 1:
                val_queries.append(record)
            else:
                train_queries.append(record)
    write_jsonl(output / "videos.jsonl", video_records)
    write_jsonl(output / "train_queries.jsonl", train_queries)
    write_jsonl(output / "val_queries.jsonl", val_queries)
    config = {
        "schema_version": 1,
        "experiment": {
            "name": "mtvr_mml_smoke_test",
            "dataset": "deterministic synthetic diagnostic data",
            "seeds": [args.seed],
            "notes": "Functional smoke test only; these metrics are not research results.",
        },
        "model": {
            "video_dim": dimension,
            "subtitle_dim": dimension,
            "query_dim": dimension,
            "proxy_dim": dimension,
            "hidden_dim": 32,
            "num_heads": 4,
            "dropout": 0.0,
            "alpha": 0.3,
            "tau": 0.07,
            "lambda_svmr": 25.0,
            "beta": 1.0,
            "boundary_kernel": 3,
        },
        "data": {
            "train_queries": "train_queries.jsonl",
            "val_queries": "val_queries.jsonl",
            "videos": "videos.jsonl",
            "expected_proxy_styles": 3,
        },
        "train": {
            "batch_size": 8,
            "epochs": 2,
            "learning_rate": 0.003,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "num_workers": 0,
            "candidate_sample_size": 8,
            "candidate_keep": 4,
            "scheduler": "constant",
            "warmup_steps": 0,
            "deterministic": True,
            "drop_last": False,
            "validate_features": True,
            "log_every_steps": 1,
            "checkpoint_every_epochs": 1,
        },
        "inference": {
            "video_batch_size": 4,
            "top_videos": args.videos,
            "spans_per_video": 5,
            "svmr_spans": 20,
            "max_span_length": 5,
        },
        "evaluation": {"ks": [1, 5, 8], "ious": [0.5, 0.7]},
    }
    atomic_write_json(output / "config.json", config)
    tracked = [
        output / "videos.jsonl",
        output / "train_queries.jsonl",
        output / "val_queries.jsonl",
        output / "config.json",
        *sorted(feature_root.rglob("*.npy")),
    ]
    atomic_write_json(
        output / "SMOKE_DATA_MANIFEST.json",
        {
            "purpose": "deterministic functional smoke test; not benchmark data",
            "generator_seed": args.seed,
            "feature_dimension": dimension,
            "files": {
                path.relative_to(output).as_posix(): sha256_file(path) for path in tracked
            },
        },
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "videos": len(video_records),
                "train_queries": len(train_queries),
                "val_queries": len(val_queries),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
