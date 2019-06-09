#!/usr/bin/env python3

'''
max-seq-size:
- 50 for sst
- 80 for snli
'''
MAX_SEQ_SIZE = 80
MAX_VOCAB_SIZE = 75000
BATCH_SIZE = 32
MAX_EPOCH = 5

QTTY_DECODER_BLOCK = 1
ATTENTION_HEADS = 1
WORD_DIM = 300  # Word embedding dimension

DROPOUT = 0.2
