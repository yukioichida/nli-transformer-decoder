#!/usr/bin/env python3
import argparse
import random

import numpy as np
import torch.nn.functional as F

from modules.custom_optimizers import OpenAIAdam
from modules.log import get_logger
from modules.model import TransformerDecoder
from modules.preprocess import *
from modules.trainer import Trainer


def run_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = get_logger(args.id)
    logger.info("Start Training {} ...".format(args.id))
    preprocess = SNLIPreProcess(device, logger, args.max_prem_size, args.max_hyp_size, args.batch_size)
    train_iter, val_iter, test_iter = preprocess.build_iterators()
    vocab = preprocess.sentence_field.vocab
    eos_vocab_index = len(vocab)
    nr_classes = len(preprocess.label_field.vocab)
    # Function that prepares the input batch for the model train
    prepare_batch_fn = preprocess.include_positional_encoding
    max_seq_size = args.max_prem_size + args.max_hyp_size + 1
    model = TransformerDecoder(len(vocab), max_seq_size, args.word_dim,
                               n_blocks=args.n_blocks, n_heads=args.n_heads, dropout=args.dropout,
                               output_dim=nr_classes, eos_token=eos_vocab_index)

    nr_optimizer_update = len(train_iter) * args.epochs
    optimizer = OpenAIAdam(model.parameters(), nr_optimizer_update)
    loss_function = F.cross_entropy

    logger.info(log_train_summary(args, sum(p.numel() for p in model.parameters())))

    trainer = Trainer(model=model, optimizer=optimizer, loss_function=loss_function, model_id=args.id,
                      prepare_batch_fn=prepare_batch_fn, device=device, logger=logger)
    trainer.train(train_iter, val_iter, test_iter, args.epochs)


def log_train_summary(args, parameters):
    """

    :return: String with training hyperparameters
    """

    return """ 
        Max Premise/Hypothesis Size: {}/{}
        Batch Size: {}
        Epochs: {}
        Decoder Blocks: {}
        Attention Heads: {}
        Word Dimension (model dimension): {}
        Model Dropout: {}
        Model Parameters: {}
    """.format(args.max_prem_size, args.max_hyp_size, args.batch_size, args.epochs, args.n_blocks,
               args.n_heads, args.word_dim, args.dropout, parameters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=str, help="Training identifier")
    parser.add_argument('--n_blocks', type=int, default=1, help="Number of transformer blocks")
    parser.add_argument('--n_heads', type=int, default=1, help="Number of attention heads used")
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--word_dim', type=int, default=30, help="Word dimensionality")
    parser.add_argument('--epochs', type=int, default=70, help="Number of epoch executed")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--dataset', type=str, default='SNLIBPE')
    parser.add_argument('--max_prem_size', type=int, default=48, help='Maximum premise length')
    parser.add_argument('--max_hyp_size', type=int, default=28, help='Maximum hypothesis length')

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    cmd_args = parser.parse_args()
    print(cmd_args)
    train_id = '{}-{}blk-{}h-{}d-{}batch'.format(cmd_args.dataset, cmd_args.n_blocks, cmd_args.n_heads,
                                                 cmd_args.word_dim, cmd_args.batch_size)
    if cmd_args.id:
        train_id = cmd_args.id + '-' + train_id
    cmd_args.id = train_id
    run_train(cmd_args)
