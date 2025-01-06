#!/usr/bin/env python3

import argparse

def process_files(src_filename, dst_filename, max_length):
    with open(src_filename, "r") as src_file, \
         open(dst_filename, "w") as dst_file:

        for line in src_file:
            elements = line.strip().split(',')
            if int(elements[2]) <= max_length:
                dst_file.write(line)

def main():
    parser = argparse.ArgumentParser(description="Filter CSV file by third column value.")
    parser.add_argument(
        "--dataset",
        default="lrs3",
        help="Dataset name"
    )
    parser.add_argument(
        "--seg-duration",
        type=int,
        default=16,
        help="Max duration (second) for each segment, (Default: 16)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Max length value"
    )
    args = parser.parse_args()
    src_filename = f"{args.dataset}_train_transcript_lengths_seg{args.seg_duration}s.csv"
    dst_filename = f"{args.dataset}_train_transcript_lengths_seg{args.seg_duration}s_0to{args.max_length}.csv"
    process_files(src_filename, dst_filename, args.max_length)

if __name__ == "__main__":
    main()
