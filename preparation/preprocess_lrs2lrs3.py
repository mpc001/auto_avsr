import argparse
import glob
import math
import os
import pickle
import shutil
import warnings

import ffmpeg
from data.data_module import AVSRDataLoader
from tqdm import tqdm
from transforms import TextTransform
from utils import save_vid_aud_txt, split_file

warnings.filterwarnings("ignore")

# Argument Parsing
parser = argparse.ArgumentParser(description="LRS2LRS3 Preprocessing")
parser.add_argument(
    "--data-dir",
    type=str,
    required=True,
    help="Directory of original dataset",
)
parser.add_argument(
    "--detector",
    type=str,
    default="retinaface",
    help="Type of face detector. (Default: retinaface)",
)
parser.add_argument(
    "--landmarks-dir",
    type=str,
    default=None,
    help="Directory of landmarks",
)
parser.add_argument(
    "--root-dir",
    type=str,
    required=True,
    help="Root directory of preprocessed dataset",
)
parser.add_argument(
    "--subset",
    type=str,
    required=True,
    help="Subset of dataset",
)
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    help="Name of dataset",
)
parser.add_argument(
    "--gpu_type",
    type=str,
    required=True,
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
    "--combine-av",
    type=lambda x: (str(x).lower() == "true"),
    default=False,
    help="Merges the audio and video components to a media file.",
)
parser.add_argument(
    "--groups",
    type=int,
    default=1,
    help="Number of threads to be used in parallel.",
)
parser.add_argument(
    "--job-index",
    type=int,
    default=0,
    help="Index to identify separate jobs (useful for parallel processing).",
)
args = parser.parse_args()

seg_duration = args.seg_duration
dataset = args.dataset
text_transform = TextTransform()

# Load Data
args.data_dir = os.path.normpath(args.data_dir)
if args.gpu_type != "cuda" or "mps":
    raise ValueError("Invalid GPU type. Valid values for gpu_type are \"cuda\" and \"mps\". ")
vid_dataloader = AVSRDataLoader(
    modality="video", detector=args.detector, convert_gray=False, gpu_type=args.gpu_type
)
aud_dataloader = AVSRDataLoader(modality="audio")

seg_vid_len = seg_duration * 25
seg_aud_len = seg_duration * 16000

# Label filename
label_filename = os.path.join(
    args.root_dir,
    "labels",
    f"{dataset}_{args.subset}_transcript_lengths_seg{seg_duration}s.csv"
    if args.groups <= 1
    else f"{dataset}_{args.subset}_transcript_lengths_seg{seg_duration}s.{args.groups}.{args.job_index}.csv",
)
os.makedirs(os.path.dirname(label_filename), exist_ok=True)
print(f"Directory {os.path.dirname(label_filename)} created")

f = open(label_filename, "w")
# Step 2, extract mouth patches from segments.
dst_vid_dir = os.path.join(
    args.root_dir, dataset, dataset + f"_video_seg{seg_duration}s"
)
dst_txt_dir = os.path.join(
    args.root_dir, dataset, dataset + f"_text_seg{seg_duration}s"
)
if dataset == "lrs3":
    if args.subset == "test":
        filenames = glob.glob(
            os.path.join(args.data_dir, args.subset, "**", "*.mp4"), recursive=True
        )
    elif args.subset == "train":
        filenames = glob.glob(
            os.path.join(args.data_dir, "trainval", "**", "*.mp4"), recursive=True
        )
        filenames.extend(
            glob.glob(
                os.path.join(args.data_dir, "pretrain", "**", "*.mp4"), recursive=True
            )
        )
        filenames.sort()
    else:
        raise NotImplementedError
elif dataset == "lrs2":
    if args.subset in ["val", "test"]:
        filenames = [
            os.path.join(args.data_dir, "main", _.split()[0] + ".mp4")
            for _ in open(
                os.path.join(os.path.dirname(args.data_dir), args.subset) + ".txt"
            )
            .read()
            .splitlines()
        ]
    elif args.subset == "train":
        filenames = [
            os.path.join(args.data_dir, "main", _.split()[0] + ".mp4")
            for _ in open(
                os.path.join(os.path.dirname(args.data_dir), args.subset) + ".txt"
            )
            .read()
            .splitlines()
        ]
        pretrain_filenames = [
            os.path.join(args.data_dir, "pretrain", _.split()[0] + ".mp4")
            for _ in open(os.path.join(os.path.dirname(args.data_dir), "pretrain.txt"))
            .read()
            .splitlines()
        ]
        filenames.extend(pretrain_filenames)
        filenames.sort()
    else:
        raise NotImplementedError

unit = math.ceil(len(filenames) * 1.0 / args.groups)
filenames = filenames[args.job_index * unit : (args.job_index + 1) * unit]

for data_filename in tqdm(filenames):
    if args.landmarks_dir:
        landmarks_filename = (
            data_filename.replace(args.data_dir, args.landmarks_dir)[:-4] + ".pkl"
        )
        landmarks = pickle.load(open(landmarks_filename, "rb"))
    else:
        landmarks = None
    try:
        video_data = vid_dataloader.load_data(data_filename, landmarks)
        audio_data = aud_dataloader.load_data(data_filename)
    except (UnboundLocalError, TypeError, OverflowError, AssertionError):
        continue

    if os.path.normpath(data_filename).split(os.sep)[-3] in [
        "trainval",
        "test",
        "main",
    ]:
        dst_vid_filename = (
            f"{data_filename.replace(args.data_dir, dst_vid_dir)[:-4]}.mp4"
        )
        dst_aud_filename = (
            f"{data_filename.replace(args.data_dir, dst_vid_dir)[:-4]}.wav"
        )
        dst_txt_filename = (
            f"{data_filename.replace(args.data_dir, dst_txt_dir)[:-4]}.txt"
        )
        trim_vid_data, trim_aud_data = video_data, audio_data
        text_line_list = (
            open(data_filename[:-4] + ".txt", "r").read().splitlines()[0].split(" ")
        )
        text_line = " ".join(text_line_list[2:])
        content = text_line.replace("}", "").replace("{", "")

        if trim_vid_data is None or trim_aud_data is None:
            continue
        video_length = len(trim_vid_data)
        audio_length = trim_aud_data.size(1)
        if video_length == 0 or audio_length == 0:
            continue
        # if audio_length/video_length < 560. or audio_length/video_length > 720. or video_length < 12:
        #    continue
        save_vid_aud_txt(
            dst_vid_filename,
            dst_aud_filename,
            dst_txt_filename,
            trim_vid_data,
            trim_aud_data,
            content,
            video_fps=25,
            audio_sample_rate=16000,
        )

        if args.combine_av:
            in1 = ffmpeg.input(dst_vid_filename)
            in2 = ffmpeg.input(dst_aud_filename)
            out = ffmpeg.output(
                in1["v"],
                in2["a"],
                dst_vid_filename[:-4] + ".av.mp4",
                vcodec="copy",
                acodec="aac",
                strict="experimental",
                loglevel="panic",
            )
            out.run()
            shutil.move(dst_vid_filename[:-4] + ".av.mp4", dst_vid_filename)

        basename = os.path.relpath(
            dst_vid_filename, start=os.path.join(args.root_dir, dataset)
        )
        token_id_str = " ".join(
            map(str, [_.item() for _ in text_transform.tokenize(content)])
        )
        f.write(
            "{}\n".format(
                f"{dataset},{basename},{trim_vid_data.shape[0]},{token_id_str}"
            )
        )
        continue

    splitted = split_file(data_filename[:-4] + ".txt", max_frames=seg_vid_len)
    for i in range(len(splitted)):
        if len(splitted) == 1:
            content, start, end, duration = splitted[i]
            trim_vid_data, trim_aud_data = video_data, audio_data
        else:
            content, start, end, duration = splitted[i]
            start_idx, end_idx = int(start * 25), int(end * 25)
            try:
                trim_vid_data, trim_aud_data = (
                    video_data[start_idx:end_idx],
                    audio_data[:, start_idx * 640 : end_idx * 640],
                )
            except TypeError:
                continue
        dst_vid_filename = (
            f"{data_filename.replace(args.data_dir, dst_vid_dir)[:-4]}_{i:02d}.mp4"
        )
        dst_aud_filename = (
            f"{data_filename.replace(args.data_dir, dst_vid_dir)[:-4]}_{i:02d}.wav"
        )
        dst_txt_filename = (
            f"{data_filename.replace(args.data_dir, dst_txt_dir)[:-4]}_{i:02d}.txt"
        )

        if trim_vid_data is None or trim_aud_data is None:
            continue
        video_length = len(trim_vid_data)
        audio_length = trim_aud_data.size(1)
        if video_length == 0 or audio_length == 0:
            continue
        save_vid_aud_txt(
            dst_vid_filename,
            dst_aud_filename,
            dst_txt_filename,
            trim_vid_data,
            trim_aud_data,
            content,
            video_fps=25,
            audio_sample_rate=16000,
        )

        if args.combine_av:
            in1 = ffmpeg.input(dst_vid_filename)
            in2 = ffmpeg.input(dst_aud_filename)
            out = ffmpeg.output(
                in1["v"],
                in2["a"],
                dst_vid_filename[:-4] + ".av.mp4",
                vcodec="copy",
                acodec="aac",
                strict="experimental",
                loglevel="panic",
            )
            out.run()
            os.remove(dst_aud_filename)
            shutil.move(dst_vid_filename[:-4] + ".av.mp4", dst_vid_filename)

        basename = os.path.relpath(
            dst_vid_filename, start=os.path.join(args.root_dir, dataset)
        )
        token_id_str = " ".join(
            map(str, [_.item() for _ in text_transform.tokenize(content)])
        )
        if token_id_str:
            f.write(
                "{}\n".format(
                    f"{dataset},{basename},{trim_vid_data.shape[0]},{token_id_str}"
                )
            )
f.close()
