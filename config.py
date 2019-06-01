#!/usr/bin/env python3

MAX_SEQ_SIZE = 200
MAX_VOCAB_SIZE = 250000
BATCH_SIZE = 32

EOS_INDEX = MAX_SEQ_SIZE  # Index of END OF SENTENCE token in embedding matrix
POS_IDX_START = EOS_INDEX + 1  # First index of position encoding in embedding matrix
POS_IDX_END = MAX_SEQ_SIZE + POS_IDX_START  # Last index of position encoding

QTTY_DECODER_BLOCK = 1
ATTENTION_HEADS = 1
WORD_DIM = 768  # Word embedding dimension

MAX_EPOCH = 20
