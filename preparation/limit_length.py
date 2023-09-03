#!/usr/bin/env python3
import os
import argparse


def process_files(src_filename, dst_filename, max_length):
    src_count, dst_count = 0, 0
    
    # Read source file
    with open(src_filename, "r") as src_file:
        src_lines = src_file.read().splitlines()
    
    # Process and write to destination file
    with open(dst_filename, "w") as dst_file:
        for line in src_lines:
            elements = line.strip().split(',')
            current_count = int(elements[2])
            src_count += current_count
            
            if current_count <= max_length:
                dst_file.write(line + '\n')
                dst_count += current_count
    
    # Print summary
    src_hours = src_count / 25.0 / 3600.0
    dst_hours = dst_count / 25.0 / 3600.0
    print(f"{os.path.basename(src_filename)}: {src_hours:.2f} hours.")
    print(f"{os.path.basename(dst_filename)}: {dst_hours:.2f} hours.")


def main():
    parser = argparse.ArgumentParser(description="Filter CSV file by third column value.")
    parser.add_argument(
        "--root-dir",
        type=str,
        required=True,
        help="Root directory of preprocessed dataset",
    )
    parser.add_argument(
        "--dataset",
        default="lrs3",
        help="Dataset name"
    )
    parser.add_argument(
        "--seg-duration",
        type=int,
        default=24,
        help="Max duration (second) for each segment, (Default: 24)",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Max length value"
    )
    args = parser.parse_args()
    src_filename = f"{args.root_dir}{args.dataset}_train_transcript_lengths_seg{args.seg_duration}s.csv"
    src_filename = os.path.join(args.root_dir, "labels", f"{args.dataset}_train_transcript_lengths_seg{args.seg_duration}s.csv")
    dst_filename = os.path.join(args.root_dir, "labels", f"{args.dataset}_train_transcript_lengths_seg{args.seg_duration}s_0to{args.max_length}.csv")
    process_files(src_filename, dst_filename, args.max_length)
    print("Done.")

if __name__ == "__main__":
    main()
