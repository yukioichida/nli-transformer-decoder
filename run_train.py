#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from preprocess import SSTPreProcess
from trainer import Trainer
from config import *
from model import TransformerDecoder
from custom_optimizers import OpenAIAdam


def run_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    preprocess = SSTPreProcess(device)
    train_iter, val_iter, test_iter = preprocess.build_iterators()
    vocab = preprocess.sentence_field.vocab
    eos_vocab_index = vocab.stoi['<eos>']
    nr_classes = len(preprocess.label_field.vocab)
    train_size = len(train_iter) * BATCH_SIZE

    prepare_batch_fn = preprocess.include_positional_encoding

    model = TransformerDecoder(len(vocab), MAX_SEQ_SIZE, WORD_DIM,
                               n_layers=QTTY_DECODER_BLOCK, n_heads=ATTENTION_HEADS, dropout=DROPOUT,
                               output_dim=nr_classes, eos_token=eos_vocab_index)
    model = model.to(device)

    nr_optimizer_update = (train_size // BATCH_SIZE) * MAX_EPOCH
    optimizer = OpenAIAdam(model.parameters(), nr_optimizer_update)
    loss_function = F.cross_entropy

    trainer = Trainer(model=model, optimizer=optimizer, loss_function=loss_function, prepare_batch_fn=prepare_batch_fn,
                      device=device)
    trainer.train(train_iter, val_iter, test_iter)


if __name__ == '__main__':
    run_train()
