from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterator

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtvr.config import MMLConfig, load_experiment_config
from mtvr.data import VideoStore, load_query_bundle, read_jsonl, write_jsonl
from mtvr.inference import predict_query
from mtvr.model import MML
from mtvr.runtime import (
    atomic_write_json,
    collect_environment,
    parameter_count,
    select_device,
    sha256_file,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-corpus MML inference (training pruning is disabled)"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val"), default="val")
    parser.add_argument("--queries", type=Path, help="override the split manifest")
    parser.add_argument("--videos", type=Path, help="override the video manifest")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--max-queries", type=int, default=0, help="diagnostic limit")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_checkpoint(path: Path, config_model: MMLConfig) -> tuple[MML, dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {"model_config", "model_state"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"checkpoint is missing fields: {missing}")
    model_config = MMLConfig.from_dict(checkpoint["model_config"])
    if model_config != config_model:
        raise ValueError(
            "checkpoint model configuration differs from --config; "
            "use the exact config that produced the checkpoint"
        )
    model = MML(model_config)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    return model, checkpoint


def main() -> None:
    args = parse_args()
    if args.max_queries < 0:
        raise ValueError("max_queries must be non-negative")
    output = args.output.resolve()
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists: {output}; pass --overwrite to replace it")

    config = load_experiment_config(args.config)
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint_path}")
    model, checkpoint = _load_checkpoint(checkpoint_path, config.model)
    device = select_device(args.device)
    model.to(device).eval()

    configured_queries = (
        config.data.train_queries if args.split == "train" else config.data.val_queries
    )
    queries_path = (
        args.queries.resolve() if args.queries else config.resolve_path(configured_queries)
    )
    videos_path = (
        args.videos.resolve() if args.videos else config.resolve_path(config.data.videos)
    )
    videos = VideoStore(
        videos_path,
        cache_size=max(128, config.inference.video_batch_size),
        model_config=config.model,
    )
    records = read_jsonl(queries_path)
    if not records:
        raise ValueError(f"query manifest is empty: {queries_path}")
    query_ids = [str(record.get("query_id", "")) for record in records]
    if len(query_ids) != len(set(query_ids)):
        raise ValueError("query manifest contains duplicate query_id values")
    if args.max_queries > 0:
        records = records[: args.max_queries]

    inference = config.inference
    processed = 0

    def generate() -> Iterator[dict[str, Any]]:
        nonlocal processed
        for index, record in enumerate(records, start=1):
            query = load_query_bundle(
                queries_path,
                record,
                model_config=config.model,
                expected_proxy_styles=config.data.expected_proxy_styles,
            )
            if query["video_id"] not in videos.by_id:
                raise ValueError(
                    f"query {query['query_id']} references unknown video {query['video_id']}"
                )
            yield predict_query(
                model,
                query,
                videos,
                video_batch_size=inference.video_batch_size,
                top_videos=inference.top_videos,
                spans_per_video=inference.spans_per_video,
                svmr_spans=inference.svmr_spans,
                max_span_length=inference.max_span_length,
                beta=model.config.beta,
                device=device,
            )
            processed = index
            if index % 100 == 0:
                print(
                    json.dumps({"processed": index, "total": len(records)}),
                    flush=True,
                )

    write_jsonl(output, generate())
    metadata = {
        "created_at_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "config": str(config.config_path),
        "config_sha256": sha256_file(config.config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_format_version": checkpoint.get("format_version"),
        "checkpoint_seed": checkpoint.get("seed"),
        "queries": str(queries_path),
        "queries_sha256": sha256_file(queries_path),
        "videos": str(videos_path),
        "videos_sha256": sha256_file(videos_path),
        "query_count": processed,
        "video_count": len(videos),
        "device": str(device),
        "parameters": parameter_count(model),
        "inference": {
            **vars(inference),
            "candidate_pruning": False,
            "full_corpus_scored": True,
        },
        "environment": collect_environment(ROOT),
        "predictions_sha256": sha256_file(output),
    }
    metadata_path = output.with_suffix(output.suffix + ".meta.json")
    atomic_write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "queries": processed,
                "videos": len(videos),
                "output": str(output),
                "metadata": str(metadata_path),
                "device": str(device),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
