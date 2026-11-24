# Generating the artistic and scribble queries

Both commands take the same selected frame. The public style configuration specifies the AnimeGANv2 checkpoint and the CLIPascene fidelity level and seed.

## Artistic

Run these commands in the SD-VCMR directory. The frame can have any positive width and height.

~~~bash
python -m pip install pillow numpy opencv-python-headless onnxruntime
mkdir -p weights
curl -fL https://raw.githubusercontent.com/TachibanaYoshino/AnimeGANv2/724b5ce75185c2e838d30c9077dbb967ab034254/pb_and_onnx_model/Shinkai_53.onnx -o weights/Shinkai_53.onnx
python scripts/style_queries.py --style artistic --image selected.png --weights weights/Shinkai_53.onnx --output artistic.png
~~~

The ONNX graph is the official Shinkai generator. Preprocessing and postprocessing follow the upstream ONNX inference script. Each input dimension below 256 becomes 256. Larger dimensions are rounded down to multiples of 32. OpenCV bilinear interpolation runs on float32 RGB before the mapping to [−1, 1]. The generator output is mapped back to [0, 255], clipped, truncated to uint8 and resized to the original frame dimensions. The release uses the ONNX Runtime CPU provider and lossless PNG storage. [Upstream inference code](https://github.com/TachibanaYoshino/AnimeGANv2/blob/724b5ce75185c2e838d30c9077dbb967ab034254/pb_and_onnx_model/test_by_onnx.py)

## Scribble

The release takes one whole-scene sketch from the official fidelity-axis procedure. It scales the frame to fit a 224 × 224 canvas with bicubic resampling and centered white padding. A single location MLP optimizes 64 one-segment cubic Bézier strokes of width 1.5. The CLIP ViT-B/32 feature loss uses zero-based block index 4 with weight 1, all other block weights 0, L2 distance and FC weight 0. The release uses seed 0, learning rate 0.0001, 1,501 iterations and four augmented views. Evaluation begins at iteration 400 and repeats every 50 iterations. Upstream inference uses the saved minimum-loss location MLP to render the strokes. The SVG is rasterized on white to produce a 224 × 224 RGB PNG. The frame stays intact during optimization, with foreground masking, foreground/background splitting and width simplification disabled. [Official fidelity-axis script](https://github.com/yael-vinker/SceneSketch/blob/2377feb466cd14fa56695b25628222e446a37b36/scripts/generate_fidelity_levels.py)

Prepare an isolated Linux environment with PyTorch, torchvision and a CUDA-enabled build of diffvg that supports your GPU. The upstream project also distributes its original environment as yaelvinker/clipascene@sha256:826e3724df0a03776185b1fbb9b24f9f66a4f271bdbdccde395d4845d4895bc7. Use the following commands from the SD-VCMR directory. For a modern PyTorch installation, apply the included compatibility patch. It uses the public padding API and moves the U2Net input tensor to the selected device.

~~~bash
ROOT="$(pwd)"
git clone https://github.com/yael-vinker/SceneSketch.git ../SceneSketch
git -C ../SceneSketch checkout 2377feb466cd14fa56695b25628222e446a37b36
git -C ../SceneSketch apply "$ROOT/docs/SceneSketch-modern-torch.patch"
mkdir -p ../SceneSketch/U2Net_/saved_models "$HOME/.cache/clip"
curl -fL https://huggingface.co/akhaliq/CLIPasso/resolve/8f40d029cb203b6cba2554779bf199f82835eb79/u2net.pth -o ../SceneSketch/U2Net_/saved_models/u2net.pth
curl -fL https://openaipublic.blob.core.windows.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt -o "$HOME/.cache/clip/ViT-B-32.pt"
python -m pip install cairosvg pillow
python scripts/style_queries.py --style scribble --image selected.png --scene-repo ../SceneSketch --output scribble.png
~~~

The optional --scene-python /path/to/python selects a separate interpreter for the optimizer, on the same filesystem. Its environment needs upstream's Python dependencies, PyTorch, OpenAI CLIP and diffvg. The wrapper needs Pillow, CairoSVG and the Cairo shared library. The optional --scene-device cpu uses a CPU build of diffvg with the compatibility patch applied, and takes substantially longer.

The configuration pins both model sources and all three weight hashes. The wrapper verifies the input weights and records the actual PNG hashes, input/output sizes and renderer details in artistic.json and scribble.json. The scribble record also includes the upstream Python diff hash and optimized SVG hash. GPU architecture, software versions and parallel rendering can affect the exact output pixels. [Style configuration](../configs/style_queries.json)

## Linux CUDA environment

The validation environment uses Ubuntu 22.04, Python 3.10, CUDA Toolkit 11.8, GCC 11 and CMake 3.22.1. The following example builds the CUDA renderer for the RTX 4060 (compute capability 8.9). Use the matching architecture for another GPU. Install the CUDA Toolkit and the Ubuntu packages git, cmake, build-essential, python3.10-dev, python3.10-venv and libcairo2 before running these commands. Keep this environment separate from LAVIS.

~~~bash
python3.10 -m venv .venv-scene
source .venv-scene/bin/activate
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install numpy==1.26.4 Pillow==9.5.0 scikit-image==0.22.0 scipy==1.12.0 \
    matplotlib==3.10.9 pandas==2.3.3 ipython==8.39.0 ftfy==6.3.1 \
    opencv-python-headless==4.11.0.86 imageio==2.38.0 wandb==0.31.2 \
    cssutils==2.15.0 svgpathtools==1.8.0 svgwrite==1.4.3 numba==0.68.1 \
    cairosvg==2.9.1 regex==2026.9.29 tqdm==4.70.1
python -m pip install --no-deps git+https://github.com/openai/CLIP.git@d05afc436d78f1c48dc0dbf8e5980a9d471f35f6

git clone https://github.com/BachiLi/diffvg.git ../diffvg
git -C ../diffvg checkout 85802a71fbcc72d79cb75716eb4da4392fd09532
git -C ../diffvg submodule update --init --recursive
cmake -S ../diffvg -B ../diffvg/build -DDIFFVG_CUDA=1 -DCMAKE_BUILD_TYPE=Release \
    -DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda -DCUDA_NVCC_FLAGS=-arch=sm_89 \
    -DPYTHON_INCLUDE_PATH="$(python -c 'import sysconfig;print(sysconfig.get_path("include"))')"
cmake --build ../diffvg/build -j3
export PYTHONPATH="$(realpath ../diffvg/build):$(realpath ../diffvg):${PYTHONPATH:-}"
python -c 'import torch, diffvg, pydiffvg; print(torch.__version__, torch.cuda.get_device_name(0))'
~~~

The compatibility patch included here has SHA-256 494427abda6d42bd17fe4af176ea6242af67cf2e65e253b5dccd3066c926e96c. The validation checkout produced this same Python diff hash. The Windows artistic validation used ONNX Runtime 1.32.0, NumPy 2.3.4, OpenCV 4.13.0 and Pillow 12.0.0. The feature extractor runs separately and receives the PNGs produced by the two style transformations.
