from __future__ import annotations

import json
import os
from collections import Counter, OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .runtime import sha256_file

if TYPE_CHECKING:
    from .config import MMLConfig


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"JSONL manifest does not exist: {source}")
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                record = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {source}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"expected an object at {source}:{line_number}")
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def resolve_manifest_path(manifest: str | Path, value: str | Path) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = Path(manifest).resolve().parent / candidate
    return candidate.resolve()


def load_feature(path: str | Path, *, expected_ndim: int | None = None) -> torch.Tensor:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"feature file does not exist: {source}")
    if source.suffix.lower() != ".npy":
        raise ValueError(f"only .npy features are supported: {source}")
    array = np.load(source, allow_pickle=False)
    if expected_ndim is not None and array.ndim != expected_ndim:
        raise ValueError(
            f"expected {expected_ndim} dimensions in {source}, got shape {array.shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"feature array must be numeric: {source}")
    if any(size <= 0 for size in array.shape):
        raise ValueError(f"feature array has an empty dimension: {source} {array.shape}")
    tensor = torch.from_numpy(np.asarray(array, dtype=np.float32))
    if not torch.isfinite(tensor).all():
        raise ValueError(f"feature array contains non-finite values: {source}")
    return tensor


class VideoStore:
    def __init__(
        self,
        manifest: str | Path,
        cache_size: int = 32,
        *,
        model_config: MMLConfig | None = None,
    ) -> None:
        self.manifest = Path(manifest).resolve()
        self.records = read_jsonl(self.manifest)
        if not self.records:
            raise ValueError(f"video manifest is empty: {self.manifest}")
        self.model_config = model_config
        self.by_id: dict[str, dict[str, Any]] = {}
        for record in self.records:
            video_id = str(record.get("video_id", ""))
            if not video_id:
                raise ValueError("each video record requires video_id")
            if video_id in self.by_id:
                raise ValueError(f"duplicate video_id: {video_id}")
            for key in ("video_feature", "subtitle_feature"):
                if key not in record:
                    raise ValueError(f"video {video_id} is missing {key}")
            if "duration" in record and float(record["duration"]) <= 0.0:
                raise ValueError(f"video {video_id} has a non-positive duration")
            if "clip_duration" in record and float(record["clip_duration"]) <= 0.0:
                raise ValueError(f"video {video_id} has a non-positive clip_duration")
            self.by_id[video_id] = record
        self.cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.records)

    @property
    def video_ids(self) -> list[str]:
        return [str(record["video_id"]) for record in self.records]

    def load(self, video_id: str) -> dict[str, Any]:
        if video_id in self._cache:
            value = self._cache.pop(video_id)
            self._cache[video_id] = value
            return value
        if video_id not in self.by_id:
            raise KeyError(f"unknown video_id: {video_id}")
        record = self.by_id[video_id]
        video = load_feature(
            resolve_manifest_path(self.manifest, record["video_feature"]),
            expected_ndim=2,
        )
        subtitle = load_feature(
            resolve_manifest_path(self.manifest, record["subtitle_feature"]),
            expected_ndim=2,
        )
        if video.shape[0] != subtitle.shape[0]:
            raise ValueError(
                f"clip-count mismatch for {video_id}: {video.shape[0]} vs {subtitle.shape[0]}"
            )
        if video.shape[0] == 0:
            raise ValueError(f"video {video_id} contains no clips")
        if self.model_config is not None:
            if video.shape[1] != self.model_config.video_dim:
                raise ValueError(
                    f"video feature width for {video_id} is {video.shape[1]}, "
                    f"expected {self.model_config.video_dim}"
                )
            if subtitle.shape[1] != self.model_config.subtitle_dim:
                raise ValueError(
                    f"subtitle feature width for {video_id} is {subtitle.shape[1]}, "
                    f"expected {self.model_config.subtitle_dim}"
                )
        value = {
            "video_id": video_id,
            "video_features": video,
            "subtitle_features": subtitle,
            "duration": record.get("duration"),
            "clip_duration": record.get("clip_duration"),
        }
        if self.cache_size:
            self._cache[video_id] = value
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return value


def load_query_bundle(
    manifest: str | Path,
    record: dict[str, Any],
    *,
    model_config: MMLConfig | None = None,
    expected_proxy_styles: int | None = None,
) -> dict[str, Any]:
    source = Path(manifest).resolve()
    query_id = str(record.get("query_id", ""))
    video_id = str(record.get("video_id", ""))
    if not query_id or not video_id:
        raise ValueError("each query record requires query_id and video_id")
    for key in ("query_feature", "proxy_feature"):
        if key not in record:
            raise ValueError(f"query {query_id} is missing {key}")
    query = load_feature(
        resolve_manifest_path(source, record["query_feature"]), expected_ndim=1
    )
    proxy = load_feature(
        resolve_manifest_path(source, record["proxy_feature"]), expected_ndim=2
    )
    if proxy.shape[0] == 0:
        raise ValueError(f"query {query_id} has no proxy styles")
    if model_config is not None:
        if query.shape[0] != model_config.query_dim:
            raise ValueError(
                f"query feature width for {query_id} is {query.shape[0]}, "
                f"expected {model_config.query_dim}"
            )
        if proxy.shape[1] != model_config.proxy_dim:
            raise ValueError(
                f"proxy feature width for {query_id} is {proxy.shape[1]}, "
                f"expected {model_config.proxy_dim}"
            )
    if expected_proxy_styles is not None and proxy.shape[0] != expected_proxy_styles:
        raise ValueError(
            f"query {query_id} has {proxy.shape[0]} proxy styles, "
            f"expected {expected_proxy_styles}"
        )
    value = {
        "query_id": query_id,
        "video_id": video_id,
        "query_features": query,
        "proxy_features": proxy,
    }
    if "start_clip" in record:
        value["start_clip"] = int(record["start_clip"])
    if "end_clip" in record:
        value["end_clip"] = int(record["end_clip"])
    return value


class MTVRDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        query_manifest: str | Path,
        video_manifest: str | Path,
        *,
        video_cache_size: int = 32,
        model_config: MMLConfig | None = None,
        expected_proxy_styles: int | None = None,
    ) -> None:
        self.query_manifest = Path(query_manifest).resolve()
        self.query_records = read_jsonl(self.query_manifest)
        if not self.query_records:
            raise ValueError(f"query manifest is empty: {self.query_manifest}")
        query_ids = [str(record.get("query_id", "")) for record in self.query_records]
        duplicates = sorted(
            query_id for query_id, count in Counter(query_ids).items() if count > 1
        )
        if duplicates:
            raise ValueError(f"duplicate query_id values: {duplicates[:10]}")
        self.model_config = model_config
        self.expected_proxy_styles = expected_proxy_styles
        self.videos = VideoStore(
            video_manifest, cache_size=video_cache_size, model_config=model_config
        )
        unknown = sorted(
            {
                str(record.get("video_id", ""))
                for record in self.query_records
                if str(record.get("video_id", "")) not in self.videos.by_id
            }
        )
        if unknown:
            raise ValueError(f"queries reference unknown video IDs: {unknown[:10]}")

    def __len__(self) -> int:
        return len(self.query_records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        query = load_query_bundle(
            self.query_manifest,
            self.query_records[index],
            model_config=self.model_config,
            expected_proxy_styles=self.expected_proxy_styles,
        )
        if "start_clip" not in query or "end_clip" not in query:
            raise ValueError(f"training query {query['query_id']} requires clip boundaries")
        video = self.videos.load(query["video_id"])
        start = int(query["start_clip"])
        end = int(query["end_clip"])
        clip_count = int(video["video_features"].shape[0])
        if not 0 <= start <= end < clip_count:
            raise ValueError(
                f"invalid span [{start}, {end}] for {query['query_id']} with {clip_count} clips"
            )
        return {**query, **video, "start_clip": start, "end_clip": end}


def pad_video_batch(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("cannot collate an empty video batch")
    max_clips = max(int(record["video_features"].shape[0]) for record in records)
    video_dim = int(records[0]["video_features"].shape[1])
    subtitle_dim = int(records[0]["subtitle_features"].shape[1])
    batch_size = len(records)
    video = torch.zeros(batch_size, max_clips, video_dim, dtype=torch.float32)
    subtitle = torch.zeros(batch_size, max_clips, subtitle_dim, dtype=torch.float32)
    mask = torch.zeros(batch_size, max_clips, dtype=torch.bool)
    for index, record in enumerate(records):
        clip_count = int(record["video_features"].shape[0])
        if record["video_features"].shape[1] != video_dim:
            raise ValueError("inconsistent video feature dimensions")
        if record["subtitle_features"].shape[1] != subtitle_dim:
            raise ValueError("inconsistent subtitle feature dimensions")
        video[index, :clip_count] = record["video_features"]
        subtitle[index, :clip_count] = record["subtitle_features"]
        mask[index, :clip_count] = True
    return {
        "video_ids": [str(record["video_id"]) for record in records],
        "video_features": video,
        "subtitle_features": subtitle,
        "clip_mask": mask,
    }


def collate_samples(records: list[dict[str, Any]]) -> dict[str, Any]:
    videos = pad_video_batch(records)
    proxy_counts = {int(record["proxy_features"].shape[0]) for record in records}
    if len(proxy_counts) != 1:
        raise ValueError("all samples in a batch must have the same number of proxy styles")
    query_dims = {int(record["query_features"].shape[0]) for record in records}
    proxy_dims = {int(record["proxy_features"].shape[1]) for record in records}
    if len(query_dims) != 1 or len(proxy_dims) != 1:
        raise ValueError("query/proxy feature dimensions must be consistent within a batch")
    return {
        **videos,
        "query_ids": [str(record["query_id"]) for record in records],
        "query_video_ids": [str(record["video_id"]) for record in records],
        "query_features": torch.stack(
            [record["query_features"] for record in records], dim=0
        ),
        "proxy_features": torch.stack(
            [record["proxy_features"] for record in records], dim=0
        ),
        "start_clip": torch.tensor(
            [int(record["start_clip"]) for record in records], dtype=torch.long
        ),
        "end_clip": torch.tensor(
            [int(record["end_clip"]) for record in records], dtype=torch.long
        ),
    }


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def validate_manifests(
    query_manifest: str | Path,
    video_manifest: str | Path,
    *,
    model_config: MMLConfig,
    expected_proxy_styles: int,
    require_labels: bool,
) -> dict[str, Any]:
    """Eagerly validate manifest structure, IDs, labels, and every feature array."""

    queries_path = Path(query_manifest).resolve()
    videos_path = Path(video_manifest).resolve()
    videos = VideoStore(videos_path, cache_size=0, model_config=model_config)
    clip_counts = []
    clip_count_by_id: dict[str, int] = {}
    for video_id in videos.video_ids:
        count = int(videos.load(video_id)["video_features"].shape[0])
        clip_counts.append(count)
        clip_count_by_id[video_id] = count

    records = read_jsonl(queries_path)
    if not records:
        raise ValueError(f"query manifest is empty: {queries_path}")
    seen: set[str] = set()
    labeled = 0
    for record in records:
        query = load_query_bundle(
            queries_path,
            record,
            model_config=model_config,
            expected_proxy_styles=expected_proxy_styles,
        )
        query_id = query["query_id"]
        if query_id in seen:
            raise ValueError(f"duplicate query_id: {query_id}")
        seen.add(query_id)
        if query["video_id"] not in videos.by_id:
            raise ValueError(
                f"query {query_id} references unknown video_id {query['video_id']}"
            )
        has_start = "start_clip" in query
        has_end = "end_clip" in query
        if has_start != has_end:
            raise ValueError(f"query {query_id} must provide both clip boundaries")
        if require_labels and not has_start:
            raise ValueError(f"query {query_id} is missing required clip boundaries")
        if has_start:
            labeled += 1
            clips = clip_count_by_id[query["video_id"]]
            start, end = int(query["start_clip"]), int(query["end_clip"])
            if not 0 <= start <= end < clips:
                raise ValueError(
                    f"invalid span [{start}, {end}] for {query_id} with {clips} clips"
                )
    return {
        "query_manifest": str(queries_path),
        "query_manifest_sha256": sha256_file(queries_path),
        "queries": len(records),
        "labeled_queries": labeled,
        "video_manifest": str(videos_path),
        "video_manifest_sha256": sha256_file(videos_path),
        "videos": len(videos),
        "minimum_clips": min(clip_counts),
        "maximum_clips": max(clip_counts),
    }
