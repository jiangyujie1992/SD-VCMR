"""Generate one benchmark style from the selected RGB frame."""

import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image, ImageOps


CONFIG = Path(__file__).resolve().parents[1] / "configs" / "style_queries.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as src:
        for chunk in iter(lambda: src.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_weight(path, expected):
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"Missing or incorrect checkpoint at {path}. Expected SHA-256 {expected}.")


def artistic(image, weights, settings):
    import cv2
    import numpy as np
    import onnxruntime as ort

    check_weight(weights, settings["sha256"])
    width, height = image.size
    stride = settings["stride"]
    size = tuple(max(settings["min_dimension"], n - n % stride) for n in (width, height))
    rgb = cv2.resize(np.asarray(image, dtype=np.float32), size, interpolation=cv2.INTER_LINEAR)
    batch = (rgb / 127.5 - 1.0)[None]
    session = ort.InferenceSession(str(weights), providers=["CPUExecutionProvider"])
    output = session.run(None, {session.get_inputs()[0].name: batch})[0][0]
    if output.shape != rgb.shape or not np.isfinite(output).all():
        raise ValueError("Invalid AnimeGANv2 output tensor")
    pixels = ((output + 1.0) / 2 * 255).clip(0, 255).astype(np.uint8)
    result = Image.fromarray(cv2.resize(pixels, (width, height), interpolation=cv2.INTER_LINEAR))
    return result, {"model_input_size": list(size), "onnxruntime": ort.__version__,
                    "opencv": cv2.__version__, "provider": session.get_providers()[0]}


def scribble(image, scene_repo, scene_python, scene_device, settings, scratch):
    import cairosvg

    repo = scene_repo.resolve()
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"],
                                       text=True).strip()
    if revision != settings["revision"]:
        raise ValueError(f"SceneSketch revision {revision} differs from the configured revision")
    check_weight(repo / settings["u2net_path"], settings["u2net_sha256"])
    probe = (
        "import json, os, platform, torch, torchvision; "
        "print(json.dumps(dict(clip=os.path.expanduser('~/.cache/clip/ViT-B-32.pt'), "
        "python=platform.python_version(), torch=torch.__version__, torchvision=torchvision.__version__, "
        "cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)))"
    )
    runtime = json.loads(subprocess.check_output([str(scene_python), "-c", probe], cwd=repo, text=True))
    check_weight(Path(runtime.pop("clip")), settings["clip_sha256"])
    if scene_device == "cuda" and runtime["gpu"] is None:
        raise ValueError("CUDA was requested, but PyTorch cannot see a GPU")
    side = settings["square_size"]
    square = ImageOps.pad(image, (side, side), method=getattr(Image, "Resampling", Image).BICUBIC, color="white")
    target = scratch / "target.png"
    square.save(target)
    layers = [0] * 12
    layers[settings["layer"]] = 1
    # Mirror the official single-fidelity run_sketch.py options, with one fixed seed.
    command = [
        str(scene_python), "painterly_rendering.py", str(target),
        "--output_dir", str(scratch), "--wandb_name", "result",
        "--use_gpu", str(int(scene_device == "cuda")), "--num_paths", str(settings["strokes"]),
        "--seed", str(settings["seed"]), "--num_iter", str(settings["num_iter"]),
        "--clip_model_name", "ViT-B/32", "--clip_fc_loss_weight", "0",
        "--clip_conv_layer_weights", ",".join(map(str, layers)),
        "--mlp_train", "1", "--lr", str(settings["lr"]),
        "--save_interval", "100", "--eval_interval", str(settings["eval_interval"]),
        "--min_eval_iter", str(settings["min_eval_iter"])
    ]
    subprocess.run(command, cwd=repo, check=True)
    svg = scratch / "result" / "best_iter.svg"
    png = cairosvg.svg2png(url=str(svg), background_color="white")
    with Image.open(io.BytesIO(png)) as rendered:
        result = rendered.convert("RGB")
    if result.getextrema()[0] == (255, 255):
        raise ValueError("SceneSketch produced a blank white canvas")
    code_diff = subprocess.check_output([
        "git", "-C", str(repo), "-c", "core.autocrlf=true", "-c", "core.filemode=false",
        "diff", "--no-ext-diff", "--no-color", "--abbrev=7", "HEAD", "--", "*.py"
    ])
    return result, {"svg_sha256": sha256(svg), "cairosvg": cairosvg.__version__,
                    "scene_device": scene_device, "runtime": runtime,
                    "upstream_diff_sha256": hashlib.sha256(code_diff).hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="Output PNG, with a sibling JSON record")
    parser.add_argument("--style", required=True, choices=["artistic", "scribble"])
    parser.add_argument("--weights", type=Path, help="Official Shinkai_53.onnx file")
    parser.add_argument("--scene-repo", type=Path, help="Pinned official SceneSketch checkout")
    parser.add_argument("--scene-python", default=sys.executable, help="Interpreter with the SceneSketch dependencies")
    parser.add_argument("--scene-device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()
    if args.output.suffix.lower() != ".png" or args.image.resolve() == args.output.resolve():
        parser.error("Use a separate .png output path")
    if args.style == "artistic" and not args.weights:
        parser.error("--weights is required for artistic")
    if args.style == "scribble" and not args.scene_repo:
        parser.error("--scene-repo is required for scribble")
    settings = json.loads(CONFIG.read_text(encoding="utf-8"))[args.style]
    input_bytes = args.image.read_bytes()
    with Image.open(io.BytesIO(input_bytes)) as decoded:
        image = decoded.convert("RGB")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output.parent.resolve()) as tmp:
        scratch = Path(tmp)
        if args.style == "artistic":
            result, details = artistic(image, args.weights, settings)
        else:
            result, details = scribble(image, args.scene_repo, args.scene_python, args.scene_device, settings, scratch)
        output = scratch / "output.png"
        result.save(output)
        metadata = {"style": args.style, "settings": settings, **details,
                    "input_size": list(image.size), "output_size": list(result.size),
                    "input_sha256": hashlib.sha256(input_bytes).hexdigest(),
                    "output_sha256": sha256(output)}
        record = scratch / "record.json"
        record.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(output, args.output)
        os.replace(record, args.output.with_suffix(".json"))


if __name__ == "__main__":
    main()
