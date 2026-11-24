"""Build the three image cues for each annotated moment in one official split."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parent


def annotations(path, dataset):
    with path.open(encoding="utf-8") as stream:
        if dataset == "tvr":
            for line in stream:
                if line.strip():
                    yield json.loads(line)
        else:
            for video_id, entry in json.load(stream).items():
                for i, (caption, interval) in enumerate(
                    zip(entry["sentences"], entry["timestamps"], strict=True)
                ):
                    yield {"vid_name": video_id, "desc_id": f"{video_id}_{i}",
                           "desc": caption, "ts": interval, "duration": entry["duration"]}


def safe_id(value):
    value = str(value)
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", value):
        raise ValueError(f"invalid ID {value!r}")
    return value


def build(args, row, work):
    def run(python, script, *arguments):
        subprocess.run([str(python), str(SCRIPTS / script), *map(str, arguments)], check=True)

    video_id = safe_id(row["vid_name"])
    qid = safe_id(row["desc_id"])
    videos = [args.videos / (video_id + ext) for ext in (".mp4", ".mkv", ".avi", ".mov", ".webm")]
    matches = [video for video in videos if video.is_file()]
    if len(matches) != 1:
        raise ValueError(f"expected one video for {video_id} in {args.videos}, found {matches}")
    annotation = work / "annotation.jsonl"
    annotation.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    local = ["--local-weights", args.local_weights, "--scratch", args.scratch] if args.local_weights else []
    run(args.select_python, "select_query_frame.py", "--annotations", annotation,
        "--video", matches[0], "--query-id", qid, "--output", work, "--device", args.select_device, *local)
    frame = work / "frame.png"
    sketch, lr, art = [work / name for name in ("scribble.png", "lowres.png", "artistic.png")]
    run(sys.executable, "visual_features.py", "lowres", frame, lr)
    run(args.style_python, "style_queries.py", "--style", "artistic", "--image", frame,
        "--output", art, "--weights", args.anime_weights)
    run(args.style_python, "style_queries.py", "--style", "scribble", "--image", frame,
        "--output", sketch, "--scene-repo", args.scene_repo, "--scene-python", args.scene_python,
        "--scene-device", args.scene_device)
    run(sys.executable, "visual_features.py", "encode", sketch, lr, art,
        "--output", work / "proxy.npy", "--device", args.feature_device)
    run(sys.executable, "visual_features.py", "encode", frame,
        "--output", work / "selected_frame_feature.npy", "--device", args.feature_device)
    return {"query_id": qid, "video_id": video_id, "query_text": row["desc"],
            "ts_seconds": row["ts"], "proxy_feature": f"{qid}/proxy.npy",
            "images": [f"{qid}/{p.name}" for p in (sketch, lr, art)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("tvr", "activitynet"))
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--videos", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="new split directory")
    parser.add_argument("--query-id", help="only process this ID")
    parser.add_argument("--anime-weights", required=True, type=Path)
    parser.add_argument("--scene-repo", required=True, type=Path)
    parser.add_argument("--select-python", default=sys.executable)
    parser.add_argument("--style-python", default=sys.executable)
    parser.add_argument("--scene-python", default=sys.executable)
    parser.add_argument("--select-device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--local-weights", type=Path, help="verified local LAVIS weights for CPU selection")
    parser.add_argument("--scratch", type=Path, help="scratch directory for the file-backed CPU model")
    parser.add_argument("--scene-device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--feature-device", default="cpu")
    args = parser.parse_args()
    if bool(args.local_weights) != bool(args.scratch) or (args.local_weights and args.select_device != "cpu"):
        parser.error("use --local-weights together with --scratch and --select-device cpu")
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        parser.error("--output must name a new directory")
    args.output.mkdir()
    pending = args.output / "visual_queries.partial.jsonl"
    count = 0
    with pending.open("x", encoding="utf-8") as manifest:
        for row in annotations(args.annotations, args.dataset):
            qid = safe_id(row["desc_id"])
            if args.query_id is not None and qid != args.query_id:
                continue
            directory = args.output / qid
            directory.mkdir()
            manifest.write(json.dumps(build(args, row, directory), ensure_ascii=False) + "\n")
            manifest.flush()
            count += 1
    if not count:
        parser.error("no matching annotated queries")
    pending.rename(args.output / "visual_queries.jsonl")


if __name__ == "__main__":
    main()
