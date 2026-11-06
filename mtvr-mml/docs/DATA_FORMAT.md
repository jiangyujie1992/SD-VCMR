# Feature and manifest contract

MML consumes frozen features and never decodes raw video during training or
inference. Manifests are UTF-8 JSON Lines files with one JSON object per line.
Relative feature paths are resolved against the directory containing that
manifest. NumPy arrays must be numeric `.npy` files readable with
`allow_pickle=False`; all values must be finite.

Run the contract checker before an experiment:

```bash
python scripts/validate_data.py --config configs/tvr_mml.json --split all
```

## Video manifest

```json
{
  "video_id": "video_00001",
  "video_feature": "features/video/video_00001.npy",
  "subtitle_feature": "features/subtitle/video_00001.npy",
  "duration": 73.5,
  "clip_duration": 1.5
}
```

| Field | Required | Contract |
|---|---:|---|
| `video_id` | yes | unique, non-empty corpus ID |
| `video_feature` | yes | float-compatible `[T, video_dim]` array |
| `subtitle_feature` | yes | float-compatible `[T, subtitle_dim]` array |
| `duration` | no | positive video duration in seconds |
| `clip_duration` | no | positive seconds per feature clip |

Video and subtitle arrays must have the same nonzero clip count `T`. For a
video without subtitles, store a zero array of shape `[T, subtitle_dim]` so the
interface remains explicit. IDs provide the deterministic final ranking tie
break.

## Query manifest

```json
{
  "query_id": "query_00001",
  "video_id": "video_00001",
  "query_feature": "features/query/query_00001.npy",
  "proxy_feature": "features/proxy/query_00001.npy",
  "start_clip": 12,
  "end_clip": 17,
  "query_text": "a person opens the door",
  "enhanced_text": "a person opens the door; a hallway and a wooden door"
}
```

| Field | Required | Contract |
|---|---:|---|
| `query_id` | yes | unique, non-empty split ID |
| `video_id` | yes | must exist in the video manifest |
| `query_feature` | yes | pooled enhanced-query `[query_dim]` array |
| `proxy_feature` | yes | style rows `[R, proxy_dim]`; paper setting `R=3` |
| `start_clip` | train/val | inclusive, zero-based start index |
| `end_clip` | train/val | inclusive, zero-based end, `start <= end < T` |
| `query_text` | no | audit-only original text |
| `enhanced_text` | no | audit-only text plus cached proxy captions |

The three proxy rows use the fixed order **sketch, low-resolution, cartoon**.
The enhanced query must be produced offline from the original text and cached
proxy captions. Neither text field is consumed by the feature-level model.

Keep train and validation query manifests separate and retain the official TVR
split membership. Test annotations are not expected locally because TVR test
labels are evaluated by the official server.

## Prediction JSONL

`scripts/infer.py` writes one record per query:

```json
{
  "query_id": "query_00001",
  "video_ranking": [{"video_id": "video_00001", "score": 0.73}],
  "moments": [{"video_id": "video_00001", "start_clip": 12, "end_clip": 17, "score": 0.21}],
  "svmr_moments": [{"video_id": "video_00001", "start_clip": 12, "end_clip": 17, "score": 0.18}],
  "ground_truth": {"video_id": "video_00001", "start_clip": 12, "end_clip": 17}
}
```

- `video_ranking` is sorted by corpus VR score.
- `moments` is the VCMR ranking from Eq. (16).
- `svmr_moments` contains spans localized inside the labeled ground-truth video.
- `ground_truth` is `null` for an unlabeled query; local evaluation requires it.

Temporal IoU interprets inclusive `[s,e]` as the half-open interval `[s,e+1)`.
The adjacent `.meta.json` sidecar binds predictions to exact input and
checkpoint SHA-256 digests.
