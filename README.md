# SD-VCMR

**Beyond Text Queries: Style-Diversified Visual Cues for Video Corpus Moment Retrieval**  
Yujie Jiang, Yunzhuo Sun, and Guoxiang Zhang

Given a text description and a visual cue, the retrieval task is to find the video and the time interval that contain the described event. TVR-SD and ActivityNet-SD use BLIP-2 to select a frame within the annotated target interval for each text query. A low-resolution image, a scribble, and an artistic rendering are all generated from that same frame, so the event stays fixed across visual styles.

## Source code

| Directory | What you can do |
| --- | --- |
| [evaluation/](evaluation/) | Score ranked predictions for video retrieval, moment localization within a given video, and moment retrieval across the corpus. |
| [mtvr-mml/](mtvr-mml/README.md) | Train and run the MTVR/MML reference implementation on precomputed features. The package includes network definitions, configuration files, and tests. |

This repository distributes source code. Training writes checkpoints to the local run directory. The source package does not include pretrained weights, SD query-construction scripts, or the eight baseline adaptations.

## Evaluate predictions

With Python 3.10 or later, run the standalone evaluator from the repository root. It uses the Python standard library.

```sh
python evaluation/evaluate.py predictions.jsonl
```

Supply one ranked prediction record per query using the [prediction JSONL format](mtvr-mml/docs/DATA_FORMAT.md#prediction-jsonl). The evaluator reports Recall@1, @10, and @100. For temporal localization it uses zero-based, inclusive clip indices and counts a hit at IoU ≥ 0.5 or ≥ 0.7 for the respective metric. Corpus moment retrieval also requires the correct video ID. This is a reference scorer from MTVR/MML. Its equivalence to the scorer used for the manuscript tables has yet to be verified.

## Benchmark access

TVR and ActivityNet-Captions provide the source annotations. The derived TVR-SD and ActivityNet-SD visual queries are used in ongoing research. For access requests, contact Yujie Jiang at [jiangyujie921228@outlook.com](mailto:jiangyujie921228@outlook.com). Requests are considered individually.
