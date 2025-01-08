import argparse
import glob
import math
import os
import re

import torch
import torchvision
import whisper
from tqdm import tqdm
from transforms import TextTransform

parser = argparse.ArgumentParser(description="Transcribe into text from media")
parser.add_argument(
    "--root-dir",
    type=str,
    required=True,
    help="Root directory of preprocessed dataset",
)
parser.add_argument(
    "--dataset",
    type=str,
    default="vox2",
    help="Name of dataset",
)
parser.add_argument(
    "--gpu_type",
    type=str,
    default="cuda",
    help="GPU type, either mps or cuda. (Default: cuda)",
)
parser.add_argument(
    "--seg-duration",
    type=int,
    default=16,
    help="Max duration (second) for each segment, (Default: 16)",
)
parser.add_argument(
    "--job-index",
    type=int,
    default=0,
    help="Index to identify separate jobs (useful for parallel processing)",
)
parser.add_argument(
    "--groups",
    type=int,
    default=1,
    help="Number of threads to be used in parallel",
)
args = parser.parse_args()

# Constants
chars_to_ignore_regex = r"[\,\?\.\!\-\;\:\"]"
dst_vid_dir = os.path.join(
    args.root_dir, args.dataset, f"{args.dataset}_video_seg{args.seg_duration}s"
)

text_transform = TextTransform()

# Load video files
all_files = sorted(glob.glob(os.path.join(dst_vid_dir, "**", "*.wav"), recursive=True))
unit = math.ceil(len(all_files) / args.groups)
files_to_process = all_files[args.job_index * unit : (args.job_index + 1) * unit]

# Label filename
label_filename = os.path.join(
    args.root_dir,
    "labels",
    f"{args.dataset}_train_transcript_lengths_seg{args.seg_duration}s.csv"
    if args.groups <= 1
    else f"{args.dataset}_train_transcript_lengths_seg{args.seg_duration}s.{args.groups}.{args.job_index}.csv",
)
os.makedirs(os.path.dirname(label_filename), exist_ok=True)
print(f"Directory {os.path.dirname(label_filename)} created")

f = open(label_filename, "w")

# Load ASR model
if args.gpu_type != "cuda" or "mps":
    raise ValueError("Invalid GPU type. Valid values for gpu_type are \"cuda\" and \"mps\". ")
model = whisper.load_model("medium.en", device=args.gpu_type)

# Transcription
for filename in tqdm(files_to_process):
    # Prepare destination filename
    dst_filename = filename.replace("_video_", "_text_")[:-4] + ".txt"
    os.makedirs(os.path.dirname(dst_filename), exist_ok=True)
    try:
        with torch.no_grad():
            result = model.transcribe(filename)
            transcript = (
                re.sub(chars_to_ignore_regex, "", result["text"])
                .upper()
                .replace("’", "'")
            )
            transcript = " ".join(transcript.split())
    except RuntimeError:
        continue

    # Write transcript to a text file
    if transcript:
        with open(dst_filename, "w") as k:
            k.write(f"{transcript}")

        trim_vid_data = torchvision.io.read_video(
            filename[:-4] + ".mp4", pts_unit="sec"
        )[0]
        basename = os.path.relpath(
            filename, start=os.path.join(args.root_dir, args.dataset)
            )[:-4]+".mp4"
        token_id_str = " ".join(
            map(str, [_.item() for _ in text_transform.tokenize(transcript)])
        )
        if token_id_str:
            f.write(
                "{}\n".format(
                    f"{args.dataset},{basename},{trim_vid_data.size(0)},{token_id_str}"
                )
            )
