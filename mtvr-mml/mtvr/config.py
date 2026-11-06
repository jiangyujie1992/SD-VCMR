from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar


T = TypeVar("T")


def _strict_kwargs(cls: type[T], values: dict[str, Any], section: str) -> dict[str, Any]:
    if not isinstance(values, dict):
        raise ValueError(f"configuration section '{section}' must be an object")
    allowed = {item.name for item in fields(cls)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {section} configuration keys: {unknown}")
    return dict(values)


@dataclass(frozen=True)
class MMLConfig:
    """Model hyperparameters corresponding to Eqs. (1)-(16) in the supplement."""

    video_dim: int
    subtitle_dim: int
    query_dim: int
    proxy_dim: int
    hidden_dim: int = 512
    num_heads: int = 8
    dropout: float = 0.1
    alpha: float = 0.3
    tau: float = 0.07
    lambda_svmr: float = 25.0
    beta: float = 1.0
    boundary_kernel: int = 3

    def __post_init__(self) -> None:
        dimensions = (
            self.video_dim,
            self.subtitle_dim,
            self.query_dim,
            self.proxy_dim,
            self.hidden_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("all feature dimensions must be positive")
        if self.num_heads <= 0 or self.hidden_dim % self.num_heads != 0:
            raise ValueError("num_heads must be positive and divide hidden_dim")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not math.isfinite(self.tau) or self.tau <= 0.0:
            raise ValueError("tau must be finite and positive")
        if not math.isfinite(self.lambda_svmr) or self.lambda_svmr < 0.0:
            raise ValueError("lambda_svmr must be finite and non-negative")
        if not math.isfinite(self.beta):
            raise ValueError("beta must be finite")
        if self.boundary_kernel <= 0 or self.boundary_kernel % 2 == 0:
            raise ValueError("boundary_kernel must be a positive odd integer")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "MMLConfig":
        return cls(**_strict_kwargs(cls, values, "model"))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DataConfig:
    train_queries: str
    val_queries: str
    videos: str
    expected_proxy_styles: int = 3

    def __post_init__(self) -> None:
        if not self.train_queries or not self.val_queries or not self.videos:
            raise ValueError("data manifest paths must be non-empty")
        if self.expected_proxy_styles <= 0:
            raise ValueError("expected_proxy_styles must be positive")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "DataConfig":
        return cls(**_strict_kwargs(cls, values, "data"))


@dataclass(frozen=True)
class TrainConfig:
    batch_size: int = 128
    epochs: int = 9
    learning_rate: float = 1.0e-4
    weight_decay: float = 0.01
    gradient_clip: float = 1.0
    num_workers: int = 4
    candidate_sample_size: int = 100
    candidate_keep: int = 10
    scheduler: str = "constant"
    warmup_steps: int = 0
    deterministic: bool = True
    drop_last: bool = True
    validate_features: bool = True
    log_every_steps: int = 50
    checkpoint_every_epochs: int = 1

    def __post_init__(self) -> None:
        integers = {
            "batch_size": self.batch_size,
            "epochs": self.epochs,
            "candidate_sample_size": self.candidate_sample_size,
            "candidate_keep": self.candidate_keep,
            "log_every_steps": self.log_every_steps,
            "checkpoint_every_epochs": self.checkpoint_every_epochs,
        }
        if any(value <= 0 for value in integers.values()):
            raise ValueError(f"training counts must be positive: {integers}")
        if self.num_workers < 0 or self.warmup_steps < 0:
            raise ValueError("num_workers and warmup_steps must be non-negative")
        if self.candidate_keep > self.candidate_sample_size:
            raise ValueError("candidate_keep cannot exceed candidate_sample_size")
        if self.candidate_sample_size > self.batch_size:
            raise ValueError("candidate_sample_size cannot exceed batch_size")
        if self.learning_rate <= 0.0 or not math.isfinite(self.learning_rate):
            raise ValueError("learning_rate must be finite and positive")
        if self.weight_decay < 0.0 or not math.isfinite(self.weight_decay):
            raise ValueError("weight_decay must be finite and non-negative")
        if self.gradient_clip < 0.0 or not math.isfinite(self.gradient_clip):
            raise ValueError("gradient_clip must be finite and non-negative")
        if self.scheduler not in {"constant", "cosine"}:
            raise ValueError("scheduler must be 'constant' or 'cosine'")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "TrainConfig":
        return cls(**_strict_kwargs(cls, values, "train"))


@dataclass(frozen=True)
class InferenceConfig:
    video_batch_size: int = 128
    top_videos: int = 100
    spans_per_video: int = 10
    svmr_spans: int = 100
    max_span_length: int | None = None

    def __post_init__(self) -> None:
        counts = (
            self.video_batch_size,
            self.top_videos,
            self.spans_per_video,
            self.svmr_spans,
        )
        if any(value <= 0 for value in counts):
            raise ValueError("inference counts must be positive")
        if self.max_span_length is not None and self.max_span_length <= 0:
            raise ValueError("max_span_length must be null or positive")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "InferenceConfig":
        return cls(**_strict_kwargs(cls, values, "inference"))


@dataclass(frozen=True)
class EvaluationConfig:
    ks: tuple[int, ...] = (1, 10, 100)
    ious: tuple[float, ...] = (0.5, 0.7)

    def __post_init__(self) -> None:
        if not self.ks or any(value <= 0 for value in self.ks):
            raise ValueError("evaluation ks must contain positive integers")
        if len(set(self.ks)) != len(self.ks):
            raise ValueError("evaluation ks must be unique")
        if not self.ious or any(not 0.0 <= value <= 1.0 for value in self.ious):
            raise ValueError("evaluation ious must lie in [0, 1]")
        if len(set(self.ious)) != len(self.ious):
            raise ValueError("evaluation ious must be unique")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "EvaluationConfig":
        kwargs = _strict_kwargs(cls, values, "evaluation")
        if "ks" in kwargs:
            kwargs["ks"] = tuple(int(value) for value in kwargs["ks"])
        if "ious" in kwargs:
            kwargs["ious"] = tuple(float(value) for value in kwargs["ious"])
        return cls(**kwargs)


@dataclass(frozen=True)
class ExperimentMetadata:
    name: str = "mtvr_mml"
    dataset: str = "MTVR (TVR official splits)"
    seeds: tuple[int, ...] = (42, 43, 44)
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.name or not self.dataset:
            raise ValueError("experiment name and dataset must be non-empty")
        if not self.seeds:
            raise ValueError("at least one experiment seed is required")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("experiment seeds must be unique")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentMetadata":
        kwargs = _strict_kwargs(cls, values, "experiment")
        if "seeds" in kwargs:
            kwargs["seeds"] = tuple(int(value) for value in kwargs["seeds"])
        return cls(**kwargs)


@dataclass(frozen=True)
class ExperimentConfig:
    schema_version: int
    experiment: ExperimentMetadata
    model: MMLConfig
    data: DataConfig
    train: TrainConfig
    inference: InferenceConfig
    evaluation: EvaluationConfig
    config_path: Path = field(repr=False, compare=False)

    @property
    def base_dir(self) -> Path:
        return self.config_path.parent

    def resolve_path(self, value: str | Path) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self.base_dir / path
        return path.resolve()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "experiment": asdict(self.experiment),
            "model": asdict(self.model),
            "data": asdict(self.data),
            "train": asdict(self.train),
            "inference": asdict(self.inference),
            "evaluation": asdict(self.evaluation),
        }


def load_experiment_config(path: str | Path) -> ExperimentConfig:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("configuration root must be an object")
    allowed = {
        "schema_version",
        "experiment",
        "model",
        "data",
        "train",
        "inference",
        "evaluation",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"unknown top-level configuration keys: {unknown}")
    required = {"model", "data", "train", "inference"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError(f"configuration is missing sections: {missing}")
    schema_version = int(payload.get("schema_version", 1))
    if schema_version != 1:
        raise ValueError(f"unsupported configuration schema_version: {schema_version}")
    return ExperimentConfig(
        schema_version=schema_version,
        experiment=ExperimentMetadata.from_dict(payload.get("experiment", {})),
        model=MMLConfig.from_dict(payload["model"]),
        data=DataConfig.from_dict(payload["data"]),
        train=TrainConfig.from_dict(payload["train"]),
        inference=InferenceConfig.from_dict(payload["inference"]),
        evaluation=EvaluationConfig.from_dict(payload.get("evaluation", {})),
        config_path=config_path,
    )


def resolve_config_path(config: ExperimentConfig, value: str | Path) -> Path:
    """Backward-compatible wrapper around :meth:`ExperimentConfig.resolve_path`."""

    return config.resolve_path(value)
