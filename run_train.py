#!/usr/bin/env python3
import sys

import torch
import torch.nn.functional as F
from modules.preprocess import SSTPreProcess
from modules.trainer import Trainer
from modules.config import *
from modules.model import TransformerDecoder
from modules.custom_optimizers import OpenAIAdam
from modules.log import get_logger


def run_train(train_id):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger(train_id)
    logger.info("Start Training {} ...".format(train_id))
    preprocess = SSTPreProcess(device, logger)
    train_iter, val_iter, test_iter = preprocess.build_iterators()
    vocab = preprocess.sentence_field.vocab
    eos_vocab_index = vocab.stoi['<eos>']
    nr_classes = len(preprocess.label_field.vocab)
    # Function that prepares the input batch for the model train
    prepare_batch_fn = preprocess.include_positional_encoding

    model = TransformerDecoder(len(vocab), MAX_SEQ_SIZE, WORD_DIM,
                               n_layers=QTTY_DECODER_BLOCK, n_heads=ATTENTION_HEADS, dropout=DROPOUT,
                               output_dim=nr_classes, eos_token=eos_vocab_index)
    nr_optimizer_update = len(train_iter) * MAX_EPOCH
    optimizer = OpenAIAdam(model.parameters(), nr_optimizer_update)
    loss_function = F.cross_entropy

    logger.info(log_train_summary())

    trainer = Trainer(model=model, optimizer=optimizer, loss_function=loss_function,
                      prepare_batch_fn=prepare_batch_fn, device=device, logger=logger)
    trainer.train(train_iter, val_iter, test_iter)


def log_train_summary():
    """

    :return: String with training hyperparameters
    """

    return """ 
        Max Sequence Size: {}
        Batch Size: {}
        Epochs: {}
        Decoder Blocks: {}
        Attention Heads: {}
        Word Dimension (model dimension): {}
        Model Dropout: {}
    """.format(MAX_SEQ_SIZE, BATCH_SIZE, MAX_EPOCH, QTTY_DECODER_BLOCK,
               ATTENTION_HEADS, WORD_DIM, DROPOUT)


if __name__ == '__main__':
    identifier = "train"
    if len(sys.argv) == 2:
        identifier = sys.argv[1]
    run_train(identifier)
