# Reconstructing visual queries

Start with one official split and a directory containing source clips named by video ID. TVR supplies query IDs, captions, video IDs, and annotated intervals in seconds as JSON Lines. ActivityNet Captions supplies a JSON object keyed by video ID with parallel sentence and timestamp arrays. The builder uses the original TVR query ID or the ActivityNet video ID followed by the zero-based caption index. Process each split into a separate output directory. [TVR annotations](https://github.com/jayleicn/TVRetrieval/tree/master/data) · [ActivityNet Captions](https://cs.stanford.edu/people/ranjaykrishna/densevid/)

### TVR video input

The TVR authors release decoded 3 fps JPEG frames. To use a released clip, run the following from a shell with FFmpeg on its path. This copies each JPEG as a video packet and assigns the one-based numbered frames timestamps starting at zero and spaced one third of a second apart. MKV stores these timestamps with millisecond precision. Source PTS from the original broadcast are unavailable in the JPEGs. [TVR frame release](https://github.com/jayleicn/TVRetrieval/tree/master/utils/video_feature)

An independently created MKV can have a different container hash because of muxer metadata. For the example below, the SHA-256 of the 257 original JPEG files concatenated in numeric order is `07bbfcf4e1d4e9ac5281676c5634a0287b328d924737ef29f2b8cfea6e3eb451`.

```sh
mkdir -p videos
ffmpeg -framerate 3 -start_number 1 \
  -i frames/castle_s01e01_seg02_clip_01/%05d.jpg -c:v copy \
  videos/castle_s01e01_seg02_clip_01.mkv
```

### Building one query or one split

Run the following commands from `SD-VCMR/`. Use separate Python environments for frame selection, style generation, and the CLIPascene renderer as needed. The driver environment requires NumPy, Pillow, torch and the matching torchvision build.

Prepare the selector and renderer using the linked setup guides. The frame selector records each candidate's timestamp and score. The renderer records the checkpoint, CLIPascene revision, strokes, layer, seed and iteration count. [Frame selection](FRAME_SELECTION.md) · [Style generation and weight downloads](STYLE_QUERIES.md) · [Style configuration](../configs/style_queries.json)

The public configuration specifies the checkpoint variants, integer rounding, sketch optimization and image preprocessing used by the builder.

```sh
python scripts/build_visual_queries.py --dataset tvr \
  --annotations tvr_train_release.jsonl --videos videos --query-id 45252 \
  --output runs/tvr-example --anime-weights /path/to/Shinkai_53.onnx \
  --scene-repo /path/to/SceneSketch --select-python /path/to/lavis/bin/python \
  --style-python /path/to/style/bin/python --scene-python /path/to/scene/bin/python
```

Omit `--query-id` to process the split. Use `--dataset activitynet --annotations train.json` for ActivityNet Captions. The filename stem must match the annotation's video ID, including its prefix. The generator writes one directory per query. Each contains the input annotation, the selected image and its candidate score trace, three style images, and the pooled feature arrays. The final `visual_queries.jsonl` indexes completed outputs. A failed run retains completed rows in `visual_queries.partial.jsonl` for inspection.

### Pixel count and visual features

The low-resolution transform divides each spatial dimension by five, rounds to the nearest integer (minimum one), then restores the original dimensions. Both resizes use Pillow bicubic interpolation. A 640 × 360 frame becomes 128 × 72, or 9,216 of the original 230,400 pixels (4%). Rounding changes the exact fraction for dimensions indivisible by five.

The release uses torchvision ResNet-152 with ImageNet-1K V1 weights and its evaluation transform: RGB, bilinear short-edge resize to 256, center crop to 224 × 224, scaling to [0,1] and ImageNet normalization. The output is the 2,048-dimensional pooled vector before the classifier. The same encoder handles source frames and rendered images. The three rows of `proxy.npy` follow the training loader's order: scribble, low-resolution, artistic. [Resolved settings](../configs/query_visual.json)

```sh
python scripts/visual_features.py lowres frame.png lowres.png
python scripts/visual_features.py encode scribble.png lowres.png artistic.png --output proxy.npy
python scripts/visual_features.py encode frame.png --output selected_frame_feature.npy
python scripts/verify_visual_query.py runs/tvr-example/45252
```

### Evaluating predictions

From the repository root, run `python evaluation/evaluate.py predictions.jsonl` on ranked outputs. Provide video IDs and inclusive, zero-based clip indices. Convert continuous annotation times to the clip grid used by the retrieval system before evaluation. The scorer computes VR, SVMR and VCMR separately, at ranks 1, 10 and 100 and temporal IoUs 0.5 and 0.7. [Prediction format](DATA_FORMAT.md#prediction-jsonl)

The visual-query manifest supplies the proxy features. The feature-level training package also expects enhanced-text, subtitle and temporal video features in its training manifests.

### Verified TVR example

For TVR train query 45252, BLIP-2 scored 11 frame candidates in [71.0, 74.4) seconds and selected the frame at 73.333 seconds (source JPEG 00221, maximum similarity 0.494543). Two CPU runs reproduced the same selection trace and PNG. The low-resolution image passed a pixel comparison with the two-step bicubic reference. The AnimeGANv2 image matched the upstream ONNX inference output pixel for pixel. CLIPascene completed 1,501 iterations on an RTX 4060 Laptop GPU and produced a 224 × 224 sketch. The three encodings formed a finite float32 array of shape [3, 2048]. The validation record includes all candidate scores, output hashes, renderer settings and the controlled standalone-scoring example. [Machine-readable validation record](tvr_45252_validation.json)
