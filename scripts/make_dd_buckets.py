#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse
import glob
import gzip
import io
import math
import multiprocessing
import os
import random
import shutil
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Generator, List, Union

import jsonlines
import numpy as np
import zstandard as zstd
from tqdm.auto import tqdm
from transformers import (
    GPTNeoXTokenizerFast,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
from webdataset import ShardWriter

# Increasing pool size results in more global shuffling, but use more memory.
SHARD_POOL_FACTOR = 20

# Number of documents per shard for each bucket. Note that bucket i has documents with length 2**i+1.
# We use smaller number of documents per shard for larger i's.
SHARD_SIZE = {i: min(2 ** (20 - i), 65536) for i in range(20)}

EOT_TOKEN = "<|endoftext|>"


def write_to_shard(chunks: List[Union[int, str]],
                   shard_writer: ShardWriter) -> None:
    """Writes a list of tokens to a shard file.

    :param chunks: A list of tokens to be written.
    :param shard_writer: A Webdataset writer object for managing the shard file.
    :return: None
    """
    for idx, chunk in enumerate(chunks):
        shard_writer.write({"__key__": f"{idx:012d}", "txt": str(chunk)})


@contextmanager
def get_item_reader(file_name: str) -> Generator[jsonlines.Reader, None, None]:
    """Creates an iterator for reading .jsonl files or Zstd-compressed .jsonl files.

    :param file_name: The path to the input data file.
    :return: A generator that yields items from the .jsonl file or zstd-compressed .jsonl file.
    """
    if file_name.endswith(".jsonl"):
        with jsonlines.open(file_name) as reader:
            yield reader
    elif file_name.endswith(".jsonl.gz"):
        with gzip.open(file_name, "rb") as f_in:
            with jsonlines.Reader(f_in) as jsonl_reader:
                yield jsonl_reader
    else:
        dctx = zstd.ZstdDecompressor()
        with open(file_name, "rb") as compressed_file:
            with dctx.stream_reader(compressed_file) as reader:
                with io.TextIOWrapper(reader, encoding="utf-8") as text_reader:
                    with jsonlines.Reader(text_reader) as jsonl_reader:
                        yield jsonl_reader


def get_binary(seq: List[Union[int, str]], min_log2: int = 8, max_log2: int = 13,
               randomize: bool = True) -> Dict[int, List[Union[int, str]]]:
    """Applies binary dataset decomposition to a document.

    :param seq: A list of tokenized documents.
    :param min_log2: The log2 of the minimum subsequence length to keep. Smaller subsequences will be ignored.
    :param max_log2: The log2 of the maximum subsequence length to keep. Larger subsequences will be further divided.
    :param randomize: If True, subsequences larger than 2**max_log2 will be divided randomly into subsequences
                      with lengths within the range determined by min_log2 and max_log2. If False, the division
                      prioritizes keeping the longest acceptable subsequences.
    :return: A dictionary `d` where `d[i]` contains the subsequences of `seq` with lengths of 2**i+1.
    """
    out_map = defaultdict(list)
    ps = 2 ** np.arange(max_log2 + 1 - min_log2)
    ps = ps / ps.sum()
    while len(seq) > 1:
        k = int(math.log2(len(seq) - 1))
        if k < min_log2:
            return out_map

        if k > max_log2:
            if randomize:
                k = np.random.choice(
                    np.arange(
                        max_log2, min_log2 - 1, -1), p=ps)
            else:
                k = min(k, max_log2)

        out_map[k].append(seq[:(1 << k) + 1])
        seq = seq[(1 << k) + 1:]
    return out_map


def tokenize_and_shard(
        file_names: List[str],
        my_id: int,
        output_dir: str,
        enc: Union[PreTrainedTokenizerFast, PreTrainedTokenizer],
        min_bucket: int,
        max_bucket: int) -> None:
    """Performs dataset-decomposition tokenize-and-shuffle using a single process.

    :param file_names: A list of input data files.
    :param my_id: The process ID for the current worker.
    :param output_dir: The path to the output directory where the sharded webdataset files will be stored.
    :param enc: A tokenizer object for tokenizing the input data.
    :param min_bucket: The index of the bucket containing the shortest sequences.
    :param max_bucket: The index of the bucket containing the longest sequences.
    :return: None
    """
    start_time = time.time()

    shard_writer = {}
    for k in range(min_bucket, max_bucket + 1):
        output_dir_k = os.path.join(output_dir, f'{k}')
        os.makedirs(output_dir_k, exist_ok=True)
        shard_writer[k] = ShardWriter(
            os.path.join(
                output_dir_k,
                "shard-%07d.tar"),
            maxcount=SHARD_SIZE[k])

    # dictionary where keys are log2 length, and values are list of sequences.
    chunks = defaultdict(list)

    num_entries = 0

    for file_name in file_names:
        with get_item_reader(file_name) as item_reader:
            for item in item_reader:
                string = item["text"]
                try:
                    tokens = enc(string).input_ids + [EOT_TOKEN]
                    token_map = get_binary(
                        tokens, min_log2=min_bucket, max_log2=max_bucket)
                    num_entries += 1
                except BaseException:
                    print("Failed to encode string.")
                    continue

                for k, v_list in token_map.items():
                    for v in v_list:
                        chunks[k].append(v)
                        if len(chunks[k]) == SHARD_POOL_FACTOR * SHARD_SIZE[k]:
                            random.shuffle(chunks[k])
                            write_to_shard(
                                chunks[k][:SHARD_SIZE[k]], shard_writer[k])
                            chunks[k] = chunks[k][SHARD_SIZE[k]:]

    total_time = time.time() - start_time
    print(
        f"Process {my_id} found {num_entries} entries in {total_time} seconds",
        flush=True,
    )

    # Write remaining shards
    for k in chunks.keys():
        random.shuffle(chunks[k])
        for i in range(0, len(chunks[k]), SHARD_SIZE[k]):
            if i + SHARD_SIZE[k] <= len(chunks[k]
                                        ):  # Do not allow partial shards
                write_to_shard(
                    chunks[k][i: i + SHARD_SIZE[k]], shard_writer[k])

    print(f"Process {my_id} Done.", flush=True)


def merge_process_dirs(
        output_dir: str,
        min_bucket: int,
        max_bucket: int,
        num_workers: int) -> None:
    """Merges multiple webdatasets into one for each bucket.

    :param output_dir: Path to a directory containing [num_workers] subdirectories. Each subdirectory is the output
                       of a single process of tokenize-and-shuffle and contains subdirectories for different buckets.
    :param min_bucket: The index of the bucket with the shortest sequences.
    :param max_bucket: The index of the bucket with the longest sequences.
    :param num_workers: The number of processes used for parallel computation.
    :return: None
    """
    process_dirs = os.listdir(output_dir)
    for k in tqdm(
        range(
            min_bucket,
            max_bucket +
            1),
        total=max_bucket -
            min_bucket +
            1):
        wds_dirs = [os.path.join(output_dir, p, f"{k}") for p in process_dirs]

        transfer_map = {}
        global_index = 0
        for i, dir in enumerate(wds_dirs):
            tarfiles = [os.path.join(dir, file) for file in os.listdir(
                dir) if file.endswith(".tar")]
            for a_tar in tarfiles:
                dir_path = os.path.join(output_dir, f"D_{k}")
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path, exist_ok=True)
                target_file = os.path.join(
                    dir_path, "shard-{:07d}.tar".format(global_index))
                global_index += 1
                transfer_map[a_tar] = target_file

        with multiprocessing.Pool(processes=num_workers) as pool:
            pool.starmap(
                shutil.move,
                [(src, transfer_map[src]) for src in transfer_map],
            )
    # Remove original subdirs
    for p in process_dirs:
        dir_path = os.path.join(output_dir, p)
        shutil.rmtree(dir_path)


def tokenize_shuffle(
        input_files: List[str],
        output_dir: str,
        num_workers: int,
        min_bucket: int,
        max_bucket: int,
) -> None:
    """Performs dataset-decomposition tokenize-and-shuffle using multiple processes.

    :param input_files: A list of input data files.
    :param output_dir: The path to the output directory.
    :param num_workers: The number of processes to use for parallel computation.
    :param min_bucket: The index of the bucket containing the shortest sequences.
    :param max_bucket: The index of the bucket containing the longest sequences.
    :return: None
    """
    input_files = [glob.glob(input_file) for input_file in input_files]
    input_files = [x for y in input_files for x in y]

    # Shuffle the input files
    random.shuffle(input_files)
    print("Number of input files = {}".format(len(input_files)))

    enc = GPTNeoXTokenizerFast.from_pretrained("EleutherAI/gpt-neox-20b")

    # assert len(input_files) % num_workers == 0
    files_per_worker = len(input_files) // num_workers
    file_groups = [input_files[x: x + files_per_worker]
                   for x in range(0, len(input_files), files_per_worker)]

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.starmap(
            tokenize_and_shard,
            [
                (
                    fg,
                    my_id,
                    os.path.join(output_dir, str(my_id)),
                    enc,
                    min_bucket,
                    max_bucket,
                )
                for my_id, fg in enumerate(file_groups)
            ],
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-files",
        type=str,
        nargs="+",
        help="Set of input data files.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Path to output directory.")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=32,
        help="Number of workers to use.")
    parser.add_argument(
        "--min-bucket",
        type=int,
        default=8,
        help="log2 of the shortest sequences.")
    parser.add_argument(
        "--max-bucket",
        type=int,
        default=13,
        help="log2 of the longest sequences.")

    args = parser.parse_args()

    tokenize_shuffle(
        args.input_files,
        args.output_dir,
        args.num_workers,
        args.min_bucket,
        args.max_bucket,
    )
    merge_process_dirs(
        args.output_dir,
        args.min_bucket,
        args.max_bucket,
        args.num_workers)
