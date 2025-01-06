#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import os

from datasets import load_dataset
from tqdm import tqdm


def main(output_dir: str, num_shards: int = 32):
    """Downloads the Wikipedia dataset.

    :param output_dir: The path to the output directory where the downloaded files will be saved.
    :param num_shards: The number of JSONL files to divide the downloaded data into.
    :return: None
    """
    os.makedirs(output_dir, exist_ok=True)
    data = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)

    for split, dataset in data.items():
        for i in tqdm(range(num_shards)):
            dataset_shard = dataset.shard(
                num_shards=num_shards, index=i, contiguous=True)
            output_file = os.path.join(
                output_dir, f"wiki_en_20220301_{split}_{i}.jsonl")
            dataset_shard.to_json(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Where to store the wikipedia .jsonl files",
    )

    args = parser.parse_args()
    main(args.output_dir)
