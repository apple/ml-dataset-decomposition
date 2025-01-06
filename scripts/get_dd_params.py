#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2024 Apple Inc. All Rights Reserved.
#

import argparse

import numpy as np
from numpy.typing import NDArray


def main(total_tokens: int,
         epochs: int,
         num_gpus: int,
         global_batch: int,
         num_workers: int,
         curriculum: NDArray,
         number_of_shards: NDArray,
         sequence_per_shard: NDArray,
         sequence_sizes: NDArray,
         batch_mult: NDArray) -> None:
    """Computes variable batch size training parameters based on the desired training hyperparameters.

    :param total_tokens: The total number of tokens to be processed during training.
    :param epochs: The number of epochs/checkpoints to save during training.
    :param num_gpus: The total number of GPUs to be used for training.
    :param global_batch: The global batch size for training.
    :param num_workers: The number of dataloader workers per GPU.
    :param curriculum: A numpy array of integers representing the probabilities of selecting a batch from each bucket.
    :param number_of_shards: A numpy array indicating the number of shards per source as defined in each bucket's manifest file.
    :param sequence_per_shard: A numpy array representing the number of sequences per shard for each source.
    :param sequence_sizes: A numpy array specifying the sizes of sequences per shard.
    :param batch_mult: A numpy array defining the ratio of each source's batch size compared to the source with the longest sequences.
    :return: None
    """
    # Number of tokens available per source/bucket:
    tokens_per_bucket = number_of_shards * sequence_per_shard * sequence_sizes

    # Ratio of number of tokens needed vs number of tokens available
    job_scale_factor = total_tokens / tokens_per_bucket.sum()

    # Number of tokens needed per source/bucket
    needed_tokens_per_bucket = job_scale_factor * tokens_per_bucket

    # Number of tokens needed per source/bucket per epoch
    needed_tokens_per_bucket_per_epoch = needed_tokens_per_bucket / epochs

    # Number of sequences needed per source/bucket per epoch
    needed_sequence_per_bucket_per_epoch = needed_tokens_per_bucket_per_epoch / \
        (sequence_sizes)

    # Number of sequences per source/bucket per epoch should be divisible with
    # the following numbers
    denom_condition1 = batch_mult * global_batch * num_workers
    denom_condition2 = sequence_per_shard * num_gpus * num_workers
    # Satisfying the second condition is sufficient
    assert np.all(denom_condition2 > denom_condition1)

    factors = needed_sequence_per_bucket_per_epoch / denom_condition2
    factors_int = np.int32(np.round(factors))

    def get_token_diff(proposed_factors):
        return total_tokens / epochs - \
            np.sum(proposed_factors * denom_condition2 * sequence_sizes)

    proposed_factors = factors_int

    index = len(factors_int) - 1

    tried_proposed_factors = set()
    while get_token_diff(proposed_factors) != 0:
        if index < 0:
            total_tokens_real = epochs * \
                (proposed_factors * denom_condition2 * sequence_sizes).sum()
            print(
                f"Cannot match requested number of tokens of {total_tokens:,}. Go with {total_tokens_real:,} instead.")
            break
        if tuple(proposed_factors) in tried_proposed_factors:
            index -= 1
        diff = get_token_diff(proposed_factors)
        tried_proposed_factors.add(tuple(proposed_factors))
        if diff < 0:
            proposed_factors[index] -= 1
        else:
            proposed_factors[index] += 1

    proposed_needed_sequence_per_bucket_per_epoch = proposed_factors * denom_condition2
    proposed_num_tokens_per_bucket = proposed_needed_sequence_per_bucket_per_epoch * sequence_sizes
    source_num_seq_per_epoch = " ".join(
        [str(int(x)) for x in proposed_needed_sequence_per_bucket_per_epoch])
    sampling_weights = proposed_num_tokens_per_bucket / \
        proposed_num_tokens_per_bucket[0] * proposed_factors[0]
    sampling_weights = sampling_weights * curriculum
    train_data_mix_weights = " ".join([str(int(x)) for x in sampling_weights])
    actual_tokens_per_epoch = (
        proposed_needed_sequence_per_bucket_per_epoch *
        sequence_sizes).sum()

    print("**** Use the following arguments:")

    print(f"--epochs {epochs}")
    print(f"--train-num-samples {actual_tokens_per_epoch}")
    print("--dataset-batch-mult " +
          " ".join([str(int(x)) for x in batch_mult]))
    print("--source-num-seq-per-epoch " + source_num_seq_per_epoch)
    print("--train-data-mix-weights " + train_data_mix_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokens",
        type=int,
        help=(
            "Total number of tokens to be seen during training."
            "Can be larger than number of available tokens as we allow repeated tokens."))
    parser.add_argument(
        "--epochs",
        type=int,
        help="Number of epochs/checkpoints to save.")
    parser.add_argument(
        "--gpus",
        type=int,
        help="Total number of GPUs to be used for training.")
    parser.add_argument(
        "--global-batch-size",
        type=int,
        help="Global batch size.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of dataloader workers per GPU.")
    parser.add_argument(
        "--number-of-shards",
        type=int,
        nargs="+",
        help="Number of shards per source (can read from manifest files).",
        default=[
            553,
            779,
            831,
            690,
            475,
            291],
    )  # Default values for the wiki example
    parser.add_argument(
        "--sequence-per-shard",
        type=int,
        nargs="+",
        help="Number of sequences per shard for each source (determined in each bucket's manifest file).",
        default=[
            4096,
            2048,
            1024,
            512,
            256,
            128],
    )  # Default values for the wiki example
    parser.add_argument(
        "--sequence_sizes",
        type=int,
        nargs="+",
        help="Size of sequences per shard.",
        default=[
            256,
            512,
            1024,
            2048,
            4096,
            8192],
    )  # Default values for the wiki example
    parser.add_argument(
        "--batch-mult",
        type=float,
        nargs="+",
        help="Ratio of each source batch vs the source with the longest sequences.",
        default=[
            32,
            16,
            8,
            4,
            2,
            1],
    )  # Default values for the wiki example
    parser.add_argument("--train-data-mix-weights", type=float, nargs="+",
                        help="List of odds to pick a batch from each bucket.",
                        default=[32, 16, 8, 4, 2, 1], )  # Pow-2 Curriculum
    args = parser.parse_args()
    print(args)
    main(
        args.tokens, args.epochs, args.gpus, args.global_batch_size, args.workers, np.array(
            args.train_data_mix_weights), np.array(
            args.number_of_shards), np.array(
                args.sequence_per_shard), np.array(
                    args.sequence_sizes), np.array(
                        args.batch_mult))
