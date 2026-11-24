# Representative-frame selection

Use the original annotation in seconds and its matching clip. For the public TVR
3 fps JPEG sequences, first pack the ordered JPEGs at 3 fps with frame 00001
at t=0. The resulting timestamps describe the published frame sequence. For
video files the sampler uses decoded presentation timestamps relative to the
video-stream start. Run the examples from the nested `SD-VCMR/` directory.
The [TVR video input example](VISUAL_QUERIES.md#tvr-video-input) shows the exact remux command.

For a labelled interval [s,e), sample target times s + k/3 strictly below e.
Match each target to the nearest decoded frame whose presentation timestamp
falls in [s,e). Prefer the earlier timestamp on a tie and remove duplicate
matches. An interval containing no frame raises an error.

Use Python 3.10, PyTorch 2.1.2, torchvision 0.16.2, NumPy 1.26.4, PyAV
12.3.0, transformers 4.26.1 and salesforce-lavis 1.0.2. LAVIS loads its
evaluation image and caption processors for the pretrain configuration
(224 × 224 bicubic resize and CLIP normalization for the image).
For each candidate take the maximum cosine between its 32 projected Q-Former
query embeddings and the projected text CLS token. On equal scores keep the
earliest candidate. The output PNG retains the decoded RGB dimensions.

    python scripts/select_query_frame.py --video videos/castle_s01e01_seg02_clip_01.mkv \
      --annotations tvr_train_release.jsonl --query-id 45252 --output runs/45252 --device cuda

The JSONL has one row per query with fields `desc_id`, `vid_name`, `desc`
and `ts`. The two entries in `ts` are start and end in **seconds** relative to
the clip. The filename stem must equal `vid_name`. The example uses query
45252 from the [official TVR training split](https://raw.githubusercontent.com/jayleicn/TVRetrieval/7e819959a1213731c6bfa8857098885c7a474e88/data/tvr_train_release.jsonl).
selection.json lists every candidate time and score, the winning zero-based
index, the source SHA-256 and the output frame SHA-256.

The LAVIS release uses the stage-one BLIP-2 pretrain model, the EVA ViT-g
vision encoder and the bert-base-uncased tokenizer. The two Salesforce
checkpoint files are listed below with the MD5 digests returned by their
Google Cloud Storage objects.

| Checkpoint | MD5 |
|------------|-----|
| [blip2_pretrained.pth](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained.pth) | fc7a61e46e52c995834dfedd5831d874 |
| [eva_vit_g.pth](https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth) | c483374fc17b72dc61e016680345a0ff |

[Official LAVIS v1.0.2 code](https://github.com/salesforce/LAVIS/tree/baad2d7c8df599d8d9b081ba2e946626eaa2dc34)

## File-backed CPU run

For a machine with limited commit memory, supply --device cpu --local-weights
WEIGHTS --scratch CACHE. The scratch directory must exist and have at least
5 GB of free space in addition to the downloaded checkpoints. The loader verifies both official checkpoint digests,
constructs the LAVIS model on the meta device, checks parameter names and shapes,
then attaches file-backed float32 tensors for inference. The score, candidate
sampler, LAVIS feature extraction methods and preprocessing are shared with the
standard command. CACHE is temporary scratch storage.

Arrange the two .pth files in WEIGHTS, with config.json and vocab.txt from
[google-bert/bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased/tree/86b5e0934494bd15c9632b12f734a8a67f723594)
in WEIGHTS/bert-base-uncased/. Their SHA-256 values are
7160e1553ad2ca51d8c1cb066be533db31826e12d173824c1bb0cb1a4f187d20 and
07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3,
respectively. The released BLIP-2 stage-one state supplies the entire Q-Former.

    python scripts/select_query_frame.py --video videos/castle_s01e01_seg02_clip_01.mkv \
      --annotations tvr_train_release.jsonl --query-id 45252 --output runs/45252 \
      --device cpu --local-weights WEIGHTS --scratch CACHE

The CPU smoke run used the [salesforce-lavis 1.0.2 PyPI wheel](https://pypi.org/project/salesforce-lavis/1.0.2/#files) (SHA-256
489782efc2d98d8e878112b2758f82ac2e6c10dfcfa087db6656465c7784d49f).
For a lightweight Windows installation, extract that wheel into a private
PYTHONPATH directory. In that copy of lavis/__init__.py remove its four eager
wildcard imports of datasets.builders, models, processors and tasks. Set
lavis/models/__init__.py to the single import
`from lavis.models.base_model import BaseModel` and lavis/processors/__init__.py
to `from lavis.processors.base_processor import BaseProcessor`. This allows
the file-backed loader to import just the extractor and BLIP processors.
A full LAVIS installation can use the same CPU command directly.

The test used Python 3.10.19, torch 2.1.2+cpu, torchvision 0.16.2+cpu, NumPy
1.26.4, Pillow 10.4.0, PyAV 12.3.0, transformers 4.26.1, timm 0.4.12,
fairscale 0.4.4, iopath 0.1.10, omegaconf 2.3.0, opencv-python-headless
4.5.5.64 and four OpenMP/MKL threads.

### Installing the selector

In a fresh Python 3.10 environment install the PyTorch wheel corresponding to
the CPU or CUDA runtime on the host. For a full LAVIS installation use the
following commands.

```sh
python -m pip install torch==2.1.2 torchvision==0.16.2
python -m pip install numpy==1.26.4 Pillow==10.4.0 av==12.3.0 transformers==4.26.1 salesforce-lavis==1.0.2
```

For the lightweight Windows environment, run the commands below in a fresh
Python 3.10 environment. Apply the three package-initializer edits above in
the extracted copy, then set `PYTHONPATH` to the absolute path of `lavis-wheel`.
For the four-thread CPU run, set `OMP_NUM_THREADS=4` and `MKL_NUM_THREADS=4`
before starting Python.

```sh
python -m pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install numpy==1.26.4 Pillow==10.4.0 av==12.3.0 transformers==4.26.1 timm==0.4.12 fairscale==0.4.4 iopath==0.1.10 omegaconf==2.3.0 pandas==2.2.3 opencv-python-headless==4.5.5.64 ftfy==6.3.1
python -m pip download --no-deps salesforce-lavis==1.0.2
python -m zipfile -e salesforce_lavis-1.0.2-py3-none-any.whl lavis-wheel
```
