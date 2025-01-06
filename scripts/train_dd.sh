#!/bin/bash

export PYTHONPATH=$PYTHONPATH:./open_lm
torchrun --nproc-per-node 8 \
        --nnodes 1 \
        --node_rank 0 \
        --max_restarts=0 \
        --rdzv_backend c10d \
        --rdzv_conf "timeout=3000,read_timeout=10000" \
        -m open_lm.main \
        --accum-freq 4 \
        --global-batch-size 64 \
        --beta1 0.9 \
        --beta2 0.95 \
        --data-key txt \
        --ffn-type swiglu \
        --fsdp \
        --fsdp-limit-all-gathers \
        --log-every-n-steps 32 \
        --lr 0.003 \
        --lr-cooldown-end 3e-5 \
        --model open_lm_1b \
        --name dd_open_lm_1b_$RANDOM \
        --precision amp_bfloat16 \
        --qk-norm  \
        --seed 42 \
        --warmup 5000 \
        --wd 0.033 \
        --workers 1 \
        --z-loss-coefficient 0.0001 \
        --ignore-parse-errors \
        --logs /mnt/open_lm_logs/ \
        --dataset-manifest  "/mnt/processed_datasets/dclm/D_8/manifest.jsonl" \
                            "/mnt/processed_datasets/dclm/D_9/manifest.jsonl" \
                            "/mnt/processed_datasets/dclm/D_10/manifest.jsonl" \
                            "/mnt/processed_datasets/dclm/D_11/manifest.jsonl" \
                            "/mnt/processed_datasets/dclm/D_12/manifest.jsonl" \
                            "/mnt/processed_datasets/dclm/D_13/manifest.jsonl" \
        --epochs 8 \
        --train-num-samples 3607101440 \
        --dataset-batch-mult 32 16 8 4 2 1 \
        --source-num-seq-per-epoch 1507328 1277952 794624 335872 137216 61440 \
        --train-data-mix-weights 1472 1248 776 328 134 60 \
        --fsdp-amp \
        --grad-clip-norm 1 \
        --attn-name xformers_attn \
        --model-norm gain_only_lp_layer_norm \
        --wandb-project-name dd_code \
        --report-to wandb \
