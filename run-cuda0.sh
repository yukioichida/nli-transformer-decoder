#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

n_heads=12
n_blocks=12
word_dim=120
batch_size=32


python3 run_train.py --n_heads=$n_heads --n_blocks=$n_blocks --word_dim=$word_dim --batch_size=$batch_size
