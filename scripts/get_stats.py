#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import os
from pathlib import Path

import jsonlines


def main(dd_dir: str) -> None:
    """Generates statistics for a directory containing a webdataset dataset.

    :param dd_dir: The path to the directory containing the webdataset dataset.
    :return: None
    """
    total_num_tokens = 0
    bucket_dir_list = sorted(
        os.listdir(dd_dir),
        key=lambda x: int(
            x.split("_")[1]))
    for bucket_dir in bucket_dir_list:
        manifest_file = os.path.join(dd_dir, bucket_dir, "manifest.jsonl")
        num_shards = 0
        num_sequences = 0
        sequence_length = 2 ** int(bucket_dir.split("_")[1])
        if os.path.exists(manifest_file):
            with jsonlines.open(manifest_file) as reader:
                for item in reader:
                    num_shards += 1
                    num_sequences += item['num_sequences']
        num_tokens = num_sequences * sequence_length
        print(
            f"{bucket_dir:<4}: # shards: {num_shards:<6,} seq-length: {sequence_length:<8,} # sequences: {num_sequences:<12,}  # tokens: {num_tokens:<12,}")
        total_num_tokens += num_tokens
    print(20 * "*")
    print(f"Total number of tokens = {total_num_tokens:,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dd-dir",
        type=Path,
        help="Path to a dataset decomposition directory.")
    args = parser.parse_args()

    main(args.dd_dir)
