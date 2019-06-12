export CUDA_VISIBLE_DEVICES=0

n_heads = 1
n_blocks = 1
word_dim = 300

max_seq_size=150

python3 run_train.py --n_heads=$n_heads --n_blocks=$n_blocks --word_dim=$word_dim --max_seq_size=$max_seq_size