"""Validate one generated query directory and print a portable result record."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory):
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    source = directory / "frame.png"
    frame_digest = sha256(source)
    candidates = selection["candidates"]
    start, end = selection["ts"]
    if not candidates or not all(start <= r["time_s"] < end and np.isfinite(r["cosine"])
                                 for r in candidates):
        raise ValueError("Invalid candidate timestamps or scores")
    winner = max(range(len(candidates)), key=lambda i: candidates[i]["cosine"])
    if (winner != selection["selected_index"] or candidates[winner]["time_s"] != selection["selected_time_s"]
            or frame_digest != selection["frame_sha256"]):
        raise ValueError("Selected frame disagrees with the selection record")
    with Image.open(source) as image:
        original = image.convert("RGB")
    w, h = original.size
    small = (max(1, (w + 2) // 5), max(1, (h + 2) // 5))
    mode = Image.Resampling.BICUBIC
    expected = original.resize(small, mode).resize((w, h), mode)
    outputs = {}
    for name in ("scribble", "lowres", "artistic"):
        path = directory / f"{name}.png"
        digest = sha256(path)
        with Image.open(path) as image:
            rgb = image.convert("RGB")
        if name == "lowres":
            if rgb.size != original.size or rgb.tobytes() != expected.tobytes():
                raise ValueError("Low-resolution pixels disagree with the bicubic reference")
        else:
            metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
            if (metadata["input_sha256"] != frame_digest or metadata["output_sha256"] != digest
                    or metadata["style"] != name or metadata["output_size"] != list(rgb.size)):
                raise ValueError(f"{name} provenance disagrees with the decoded image")
        if all(lo == hi for lo, hi in rgb.getextrema()):
            raise ValueError(f"{name} produced a single-color image")
        if rgb.size == original.size and rgb.tobytes() == original.tobytes():
            raise ValueError(f"{name} produced an exact copy of the source")
        outputs[name] = {"size": list(rgb.size), "sha256": digest}
    for name, rows in (("proxy", 3), ("selected_frame_feature", 1)):
        path = directory / f"{name}.npy"
        feats = np.load(path, allow_pickle=False)
        if feats.dtype != np.float32 or feats.shape != (rows, 2048) or not np.isfinite(feats).all():
            raise ValueError(f"Invalid {name} features")
        outputs[name] = {"shape": list(feats.shape), "sha256": sha256(path)}
    return {"query_id": selection["desc_id"], "video_id": selection["vid_name"],
            "interval_s": selection["ts"], "candidates": len(candidates), "selected": candidates[winner],
            "input_size": [w, h], "downsampled_size": list(small), "frame_sha256": frame_digest,
            "outputs": outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    print(json.dumps(verify(parser.parse_args().directory), indent=2))
