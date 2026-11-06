# Method-to-code map

This map makes the supplement's mathematical description auditable against the
released implementation. Equation numbers refer to the full method section in
the supplementary material.

| Paper component | Equation | Implementation | Test evidence |
|---|---:|---|---|
| Enhanced semantic query projection | (1) | `MML.encode_query`, `query_projector` | `test_forward_loss_and_backward` |
| Mean-pooled proxy/style representation | (2) | `MML.encode_query`, `proxy_projector` | model and CLI tests |
| Video/subtitle feature projections | (3)-(4) | `MML.encode_video` | data dimension and model tests |
| Bidirectional video-subtitle attention | (5)-(6) | `MML.contextualize_video` | forward/backward test |
| Spatial and semantic cosine scores | (7)-(8) | `MML.localize_contextualized_pairs` | masked-logit assertions |
| Late fusion `(1-alpha) spatial + alpha semantic` | (9) | `MML.localize_contextualized_pairs` | finite-loss tests |
| Bias-free start/end Conv1D heads | (10)-(11) | `start_head`, `end_head` | boundary shape/mask tests |
| SVMR cross-entropy | (12) | `losses.boundary_loss` | backward tests |
| Max-pooled two-branch VR score | (13) | `MML.score_videos_from_embeddings` | corpus inference test |
| In-batch InfoNCE | (14) | `losses.multi_positive_info_nce` | duplicate-positive test |
| Joint objective | (15) | `losses.compute_mml_loss` | backward test |
| VCMR score | (16) | `inference.predict_query` | inference/metric test |

## Training-only spatial pruning

`MML.forward_pruned` is the actual training entry point used by
`scripts/train.py`:

1. The shuffled DataLoader batch is the in-batch candidate pool of Eq. (14).
2. `sample_candidate_indices` draws `candidate_sample_size` unique candidates
   per query and always includes its paired video.
3. Raw projected video clips and the mean-pooled proxy vector produce the cheap
   score `max_t cosine(f_v,t, F_m)` without cross-attention.
4. `retain_top_candidates` keeps the configured hard top-k and forcibly
   restores the positive if it fell below the cut.
5. Video/subtitle cross-attention and boundary heads run only on retained
   query-candidate pairs.
6. Unretained entries are `-inf` in the InfoNCE matrix, so they have no gradient.
   Multiple batch entries for the same video remain valid positives.
7. Boundary loss is computed from the forcibly retained paired video.

With `configs/tvr_mml.json`, the intended full batch is 128, 100 candidates are
sampled, and 10 are retained. The final partial batch is dropped to preserve
that protocol. The smoke configuration deliberately uses 8-to-4 pruning.

## Inference differs intentionally

`score_corpus` iterates over every video and calls the contextualized Eq. (13)
score. No training candidate mask is used. Only after full-corpus VR ranking
does `predict_query` select the configured top videos for boundary decoding.
Candidate localization is chunked by `video_batch_size` to bound memory without
changing scores.

## Frozen-feature boundary

The four `FeatureProjector` modules are trainable adapters over cached features;
the upstream ResNet, SlowFast, RoBERTa, BLIP2, and MiniGPT-v2 models are not
part of this online graph. This separation is enforced by the `.npy` data
contract rather than hidden runtime downloads.
