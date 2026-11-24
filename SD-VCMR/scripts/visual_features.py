"""Build the bicubic query and encode arbitrary RGB frames with one frozen backbone."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs" / "query_visual.json")
                    .read_text(encoding="utf-8"))


def low_resolution(image):
    rgb = image.convert("RGB")
    width, height = rgb.size
    # Round each dimension to its nearest fifth, keeping tiny images nonempty.
    divisor = CONFIG["low_resolution"]["dimension_divisor"]
    small_size = tuple(max(1, (n + divisor // 2) // divisor) for n in (width, height))
    small = rgb.resize(small_size, Image.Resampling.BICUBIC)
    return small.resize(rgb.size, Image.Resampling.BICUBIC)


class ResNetEncoder:
    def __init__(self, device):
        import torch
        from torchvision.models import ResNet152_Weights, resnet152

        weights = ResNet152_Weights[CONFIG["encoder"]["weights"]]
        self.model = resnet152(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model.to(device).eval()
        self.preprocess = weights.transforms()
        self.device = device

    def encode(self, paths):
        import torch

        frames = []
        for path in paths:
            with Image.open(path) as image:
                frames.append(self.preprocess(image.convert("RGB")))
        with torch.inference_mode():
            features = self.model(torch.stack(frames).to(self.device)).cpu().numpy()
        if features.shape != (len(paths), CONFIG["encoder"]["dimension"]) or not np.isfinite(features).all():
            raise ValueError("invalid ResNet-152 output")
        return features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    lr = sub.add_parser("lowres", help="downsample then upsample with bicubic interpolation")
    lr.add_argument("image", type=Path)
    lr.add_argument("output", type=Path, help="output PNG")
    features = sub.add_parser("encode", help="write float32 rows in input order")
    features.add_argument("images", nargs="+", type=Path)
    features.add_argument("--output", required=True, type=Path)
    features.add_argument("--device", default="cpu")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "lowres":
        with Image.open(args.image) as image:
            low_resolution(image).save(args.output, format="PNG")
    else:
        encoder = ResNetEncoder(args.device)
        rows = [encoder.encode(args.images[i:i + 8]) for i in range(0, len(args.images), 8)]
        with args.output.open("wb") as stream:
            np.save(stream, np.concatenate(rows), allow_pickle=False)


if __name__ == "__main__":
    main()
