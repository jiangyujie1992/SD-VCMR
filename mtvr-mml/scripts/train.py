from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mtvr.config import ExperimentConfig, load_experiment_config
from mtvr.data import (
    MTVRDataset,
    collate_samples,
    move_batch_to_device,
    validate_manifests,
)
from mtvr.losses import compute_mml_loss
from mtvr.model import MML
from mtvr.runtime import (
    append_jsonl,
    atomic_write_json,
    collect_environment,
    parameter_count,
    seed_everything,
    seed_worker,
    select_device,
    sha256_file,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MML with training-only spatial candidate pruning"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seed",
        type=int,
        help="run seed (defaults to the first seed listed in the config)",
    )
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--max-steps", type=int, default=0, help="smoke-test limit")
    parser.add_argument("--resume", type=Path, help="resume after a saved epoch")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="override train.deterministic",
    )
    parser.add_argument(
        "--validate-features",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="eagerly validate every configured feature before training",
    )
    return parser.parse_args()


def _numpy_state() -> dict[str, Any]:
    name, state, position, has_gauss, cached = np.random.get_state()
    return {
        "name": name,
        "state": state.tolist(),
        "position": position,
        "has_gauss": has_gauss,
        "cached_gaussian": cached,
    }


def _restore_numpy_state(payload: dict[str, Any]) -> None:
    np.random.set_state(
        (
            payload["name"],
            np.asarray(payload["state"], dtype=np.uint32),
            int(payload["position"]),
            int(payload["has_gauss"]),
            float(payload["cached_gaussian"]),
        )
    )


def save_checkpoint(
    path: Path,
    model: MML,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loader_generator: torch.Generator,
    candidate_generator: torch.Generator,
    *,
    config: ExperimentConfig,
    epoch: int,
    step: int,
    seed: int,
    metrics: dict[str, float],
) -> None:
    payload: dict[str, Any] = {
        "format_version": 2,
        "model_config": model.config.to_dict(),
        "experiment_config": config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": epoch,
        "step": step,
        "seed": seed,
        "metrics": metrics,
        "rng_state": {
            "python": random.getstate(),
            "numpy": _numpy_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "loader_generator": loader_generator.get_state(),
            "candidate_generator": candidate_generator.get_state(),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    model: MML,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    loader_generator: torch.Generator,
    candidate_generator: torch.Generator,
    config: ExperimentConfig,
) -> tuple[int, int, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 2:
        raise ValueError("resume requires a format_version=2 checkpoint")
    if checkpoint.get("model_config") != config.model.to_dict():
        raise ValueError("checkpoint model configuration does not match --config")
    if checkpoint.get("experiment_config") != config.to_dict():
        raise ValueError("checkpoint experiment configuration does not match --config")
    model.load_state_dict(checkpoint["model_state"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    scheduler.load_state_dict(checkpoint["scheduler_state"])
    rng = checkpoint["rng_state"]
    random.setstate(rng["python"])
    _restore_numpy_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if torch.cuda.is_available() and rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])
    loader_generator.set_state(rng["loader_generator"])
    candidate_generator.set_state(rng["candidate_generator"])
    return int(checkpoint["epoch"]) + 1, int(checkpoint["step"]), int(
        checkpoint["seed"]
    )


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    *,
    name: str,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def scale(step: int) -> float:
        completed = step + 1
        if warmup_steps and completed <= warmup_steps:
            return completed / warmup_steps
        if name == "constant":
            return 1.0
        progress = (completed - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def main() -> None:
    args = parse_args()
    if args.max_steps < 0:
        raise ValueError("max_steps must be non-negative")
    config = load_experiment_config(args.config)
    train_config = config.train
    seed = config.experiment.seeds[0] if args.seed is None else args.seed
    deterministic = (
        train_config.deterministic
        if args.deterministic is None
        else args.deterministic
    )
    validate_features = (
        train_config.validate_features
        if args.validate_features is None
        else args.validate_features
    )
    seed_everything(seed, deterministic=deterministic)

    queries = config.resolve_path(config.data.train_queries)
    val_queries = config.resolve_path(config.data.val_queries)
    videos = config.resolve_path(config.data.videos)
    validation: dict[str, Any] = {}
    if validate_features:
        validation["train"] = validate_manifests(
            queries,
            videos,
            model_config=config.model,
            expected_proxy_styles=config.data.expected_proxy_styles,
            require_labels=True,
        )
        validation["validation"] = validate_manifests(
            val_queries,
            videos,
            model_config=config.model,
            expected_proxy_styles=config.data.expected_proxy_styles,
            require_labels=True,
        )

    dataset = MTVRDataset(
        queries,
        videos,
        model_config=config.model,
        expected_proxy_styles=config.data.expected_proxy_styles,
    )
    if train_config.drop_last and len(dataset) < train_config.batch_size:
        raise ValueError(
            "drop_last=true requires at least batch_size training queries; "
            "set train.drop_last=false for a small diagnostic dataset"
        )
    loader_generator = torch.Generator().manual_seed(seed)
    candidate_generator = torch.Generator().manual_seed(seed + 1)
    loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=train_config.num_workers,
        collate_fn=collate_samples,
        generator=loader_generator,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        persistent_workers=train_config.num_workers > 0,
        drop_last=train_config.drop_last,
    )
    if len(loader) == 0:
        raise ValueError("training data loader contains no batches")

    device = select_device(args.device)
    model = MML(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    total_steps = train_config.epochs * len(loader)
    scheduler = make_scheduler(
        optimizer,
        name=train_config.scheduler,
        warmup_steps=train_config.warmup_steps,
        total_steps=total_steps,
    )
    output = args.output.resolve()
    if not args.resume and (output / "checkpoint_last.pt").exists():
        raise FileExistsError(
            f"refusing to overwrite an existing run: {output}; use --resume"
        )
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_epoch, step = 1, 0
    if args.resume:
        start_epoch, step, checkpoint_seed = load_checkpoint(
            args.resume.resolve(),
            model,
            optimizer,
            scheduler,
            loader_generator,
            candidate_generator,
            config,
        )
        if checkpoint_seed != seed:
            raise ValueError(
                f"resume seed {checkpoint_seed} does not match requested seed {seed}"
            )
        if args.max_steps and step >= args.max_steps:
            raise ValueError(
                f"resume checkpoint is already at step {step}, which meets "
                f"--max-steps {args.max_steps}"
            )

    metadata = {
        "status": "running",
        "started_at_utc": utc_now(),
        "command": [sys.executable, *sys.argv],
        "config_path": str(config.config_path),
        "config_sha256": sha256_file(config.config_path),
        "resolved_config": config.to_dict(),
        "resolved_data": {
            "train_queries": str(queries),
            "val_queries": str(val_queries),
            "videos": str(videos),
        },
        "seed": seed,
        "deterministic": deterministic,
        "device": str(device),
        "environment": collect_environment(ROOT),
        "parameters": parameter_count(model),
        "feature_validation": validation,
        "candidate_protocol": {
            "configured_sample_size": train_config.candidate_sample_size,
            "configured_keep": train_config.candidate_keep,
            "ground_truth_forced": True,
            "inference_pruning": False,
        },
    }
    atomic_write_json(output / "run_metadata.json", metadata)
    atomic_write_json(output / "config.resolved.json", config.to_dict())
    metrics_path = output / "metrics.jsonl"

    stop = False
    summary: dict[str, Any] = {}
    for epoch in range(start_epoch, train_config.epochs + 1):
        model.train()
        totals = {"loss": 0.0, "loss_vr": 0.0, "loss_svmr": 0.0}
        epoch_steps = 0
        for batch in loader:
            batch = move_batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model.forward_pruned(
                batch,
                sample_size=train_config.candidate_sample_size,
                keep=train_config.candidate_keep,
                generator=candidate_generator,
            )
            losses = compute_mml_loss(
                outputs,
                batch,
                tau=model.config.tau,
                lambda_svmr=model.config.lambda_svmr,
            )
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(f"non-finite loss at training step {step + 1}")
            losses["loss"].backward()
            if train_config.gradient_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    train_config.gradient_clip,
                    error_if_nonfinite=True,
                )
            optimizer.step()
            scheduler.step()
            step += 1
            epoch_steps += 1
            for key in totals:
                totals[key] += float(losses[key].detach())
            if step % train_config.log_every_steps == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "loss": round(float(losses["loss"].detach()), 6),
                            "learning_rate": optimizer.param_groups[0]["lr"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if args.max_steps and step >= args.max_steps:
                stop = True
                break

        metrics = {key: value / max(epoch_steps, 1) for key, value in totals.items()}
        summary = {
            "epoch": epoch,
            "step": step,
            "epoch_steps": epoch_steps,
            "device": str(device),
            "learning_rate": optimizer.param_groups[0]["lr"],
            "effective_candidate_sample_size": min(
                train_config.candidate_sample_size, train_config.batch_size
            ),
            "effective_candidate_keep": min(
                train_config.candidate_keep,
                train_config.candidate_sample_size,
                train_config.batch_size,
            ),
            **{key: round(value, 8) for key, value in metrics.items()},
        }
        append_jsonl(metrics_path, summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
        save_checkpoint(
            output / "checkpoint_last.pt",
            model,
            optimizer,
            scheduler,
            loader_generator,
            candidate_generator,
            config=config,
            epoch=epoch,
            step=step,
            seed=seed,
            metrics=metrics,
        )
        if epoch % train_config.checkpoint_every_epochs == 0:
            save_checkpoint(
                checkpoint_dir / f"checkpoint_epoch_{epoch:03d}.pt",
                model,
                optimizer,
                scheduler,
                loader_generator,
                candidate_generator,
                config=config,
                epoch=epoch,
                step=step,
                seed=seed,
                metrics=metrics,
            )
        if stop:
            break

    if not summary:
        raise RuntimeError("no training epoch was executed")
    summary["completed_at_utc"] = utc_now()
    summary["stopped_by_max_steps"] = stop
    atomic_write_json(output / "train_summary.json", summary)
    metadata.update(
        {
            "status": "completed",
            "completed_at_utc": summary["completed_at_utc"],
            "final_checkpoint": str((output / "checkpoint_last.pt").resolve()),
            "final_summary": summary,
        }
    )
    atomic_write_json(output / "run_metadata.json", metadata)


if __name__ == "__main__":
    main()
