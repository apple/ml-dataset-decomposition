#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import os

from datasets import load_dataset
from tqdm import tqdm


def main(output_dir: str, num_shards: int = 128) -> None:
    """Downloads a small subset of the DCLM-Baseline dataset.

    :param output_dir: The path to the output directory where the downloaded files will be saved.
    :param num_shards: The number of JSONL files to divide the downloaded data into.
    :return: None
    """
    os.makedirs(output_dir, exist_ok=True)
    # Download a small fraction of DCLM-Baseline dataset
    data = load_dataset("mlfoundations/dclm-baseline-1.0",
                        data_dir="global-shard_01_of_10/local-shard_0_of_10")

    for split, dataset in data.items():
        for i in tqdm(range(num_shards)):
            dataset_shard = dataset.shard(
                num_shards=num_shards, index=i, contiguous=True)
            output_file = os.path.join(output_dir, f"dclm_{split}_{i}.jsonl")
            dataset_shard.to_json(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to store the DCLM subset .jsonl files.",
    )

    args = parser.parse_args()
    main(args.output_dir)
