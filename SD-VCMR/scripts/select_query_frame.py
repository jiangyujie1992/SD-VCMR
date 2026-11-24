"""Select a query frame using the LAVIS BLIP-2 stage-one contrastive features."""
import argparse
from bisect import bisect_left
from fractions import Fraction
import hashlib
import json
from pathlib import Path


def nearest_grid_indices(times, start, stop):
    """Nearest in-window presentation timestamp for each 3 Hz tick, ties earlier."""
    indices = []
    tick = start
    while tick < stop:
        right = bisect_left(times, tick)
        neighbors = range(max(0, right - 1), min(len(times), right + 1))
        chosen = min(neighbors, key=lambda i: (abs(times[i] - tick), i))
        if not indices or indices[-1] != chosen:
            indices.append(chosen)
        tick += Fraction(1, 3)
    return indices


def decode_candidates(path, start, stop):
    import av

    # First pass collects timestamps only. Decode again to retain only the
    # sampled frames, even when a high-frame-rate video has a long annotation.
    def frames():
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            origin = (stream.start_time or 0) * stream.time_base
            for frame in container.decode(stream):
                if frame.pts is None:
                    raise ValueError("Decoded frame lacks a presentation timestamp")
                timestamp = frame.pts * frame.time_base - origin
                if timestamp >= stop:
                    break
                if timestamp >= start:
                    yield timestamp, frame
    times = [time for time, _ in frames()]
    if not times:
        raise ValueError(f"No presentation timestamp in [{float(start)}, {float(stop)})")
    if any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("Video has non-increasing presentation timestamps")
    chosen = set(nearest_grid_indices(times, start, stop))
    for i, (timestamp, frame) in enumerate(frames()):
        if i in chosen:
            yield timestamp, frame.to_image().convert("RGB")


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FrameSelector:
    def __init__(self, device, local_weights=None, scratch=None):
        import torch
        if local_weights is not None:
            from tempfile import TemporaryDirectory
            if __package__:
                from .select_query_frame_cpu import load_cpu_model
            else:
                from select_query_frame_cpu import load_cpu_model

            if str(device) != "cpu":
                raise ValueError("--local-weights requires --device cpu")
            self.device = torch.device(device)
            self._scratch = TemporaryDirectory(dir=scratch or local_weights)
            self.model, self.visual_processor, self.text_processor = load_cpu_model(
                local_weights, self._scratch.name
            )
            return
        from lavis.models import load_model_and_preprocess

        self.device = torch.device(device)
        self.model, vision, text = load_model_and_preprocess(
            name="blip2_feature_extractor", model_type="pretrain",
            is_eval=True, device=self.device,
        )
        self.visual_processor = vision["eval"]
        self.text_processor = text["eval"]

    def close(self):
        if hasattr(self, "_scratch"):
            del self.model
            self._scratch.cleanup()
            del self._scratch

    def select(self, video, record, output):
        import torch

        video, output = Path(video), Path(output)
        if video.stem != str(record["vid_name"]):
            raise ValueError("Video filename stem must match annotation vid_name")
        start, stop = (Fraction(str(i)) for i in record["ts"])
        if start < 0 or stop <= start:
            raise ValueError(f"Invalid annotation interval {record['ts']}")
        if not isinstance(record["desc"], str) or not record["desc"].strip():
            raise ValueError("Annotation must include a nonempty desc")
        candidates = []
        best = None
        with torch.inference_mode():
            processed_text = self.text_processor(record["desc"])
            features = self.model.extract_features({"text_input": [processed_text]}, mode="text")
            # LAVIS image-text contrastive scoring uses the text CLS embedding.
            cls = features.text_embeds_proj[0, 0].float()
            for timestamp, image in decode_candidates(video, start, stop):
                tensor = self.visual_processor(image).unsqueeze(0).to(self.device)
                features = self.model.extract_features({"image": tensor}, mode="image")
                queries = features.image_embeds_proj[0].float()
                score = (queries @ cls).max().item()
                if not -1.0001 <= score <= 1.0001:
                    raise ValueError(f"Invalid contrastive score {score}")
                candidates.append({"time_s": float(timestamp), "cosine": score})
                if best is None or score > best[0]:
                    best = score, len(candidates) - 1, image
        if best is None:
            raise ValueError("No candidate frame was scored")
        output.mkdir(parents=True, exist_ok=True)
        frame_path = output / "frame.png"
        best[2].save(frame_path)
        result = {
            "vid_name": record["vid_name"], "desc_id": record["desc_id"],
            "desc": record["desc"], "ts": record["ts"], "sampling_fps": 3,
            "model": "blip2_feature_extractor/pretrain", "device": str(self.device),
            "initialization": "file-backed" if hasattr(self, "_scratch") else "lavis",
            "model_state_tensors": len(self.model.state_dict()),
            "candidates": candidates, "selected_index": best[1],
            "selected_time_s": candidates[best[1]]["time_s"],
            "source_sha256": file_sha256(video), "frame_sha256": file_sha256(frame_path),
        }
        (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--query-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--local-weights", type=Path, help="Use verified local weights and a file-backed CPU model")
    parser.add_argument("--scratch", type=Path, help="Directory with 5 GB for CPU model storage")
    args = parser.parse_args()
    with args.annotations.open(encoding="utf-8") as stream:
        matches = [row for row in map(json.loads, filter(str.strip, stream))
                   if str(row["desc_id"]) == args.query_id]
    if len(matches) != 1:
        parser.error(f"Expected exactly one matching annotation, found {len(matches)}")
    selector = FrameSelector(args.device, args.local_weights, args.scratch)
    try:
        result = selector.select(args.video, matches[0], args.output)
    finally:
        selector.close()
    print(json.dumps({"selected_time_s": result["selected_time_s"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
