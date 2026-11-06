# SD-VCMR

This repository is the home for the code accompanying **Beyond Text Queries: Style-Diversified Visual Cues for Video Corpus Moment Retrieval**.

**Authors:** Yujie Jiang, Yunzhuo Sun, and Guoxiang Zhang

## About the study

Given a query, the system searches a collection of videos and returns the video and time interval that best match the event. We study how a visual cue can complement a short text description when selecting the video and locating the event within it.

The TVR-SD and ActivityNet-SD benchmarks pair text with three visual query styles: low-resolution images, scribbles, and artistic images. For each annotated event, the construction procedure selects a representative frame within the target interval and generates the three visual variants from that frame. This lets us compare query styles while holding the depicted event fixed.

## Code availability

The [complete MTVR/MML reference source](mtvr-mml/README.md) contains the feature-level MML model, training and inference scripts, retrieval metrics, configuration, documentation, and tests. It takes precomputed features as input. The derived TVR-SD and ActivityNet-SD query datasets, their construction scripts, and the eight SD baseline adaptations are outside the scope of this package.

The [standalone reference evaluator](evaluation/evaluate.py) scores ranked predictions for video retrieval (VR), localization within the target video (SVMR, called SD-VMR in the paper), and corpus moment retrieval (VCMR). The [metric implementation](evaluation/retrieval_metrics.py) comes from the MTVR/MML research artifact and retains its [MIT license](evaluation/LICENSE). Equivalence with the evaluator used for the table values reported in the manuscript has yet to be verified.

From the repository root, run `python evaluation/evaluate.py predictions.jsonl` with Python 3.10 or later. The input contains one JSON object per query, with predictions ranked best first. The following synthetic example shows a complete row:

```json
{"query_id":"q1","ground_truth":{"video_id":"v1","start_clip":2,"end_clip":5},"video_ranking":[{"video_id":"v1"}],"moments":[{"video_id":"v1","start_clip":2,"end_clip":3}],"svmr_moments":[{"video_id":"v1","start_clip":2,"end_clip":5}]}
```

The evaluator uses zero-based, inclusive clip indices and counts a temporal match when IoU is **greater than or equal to** the threshold (0.5 or 0.7). It reports Recall@1, @10, and @100 as percentages over the supplied queries. In the example, the corpus prediction has IoU 0.5 and counts as a hit only at the lower threshold.
