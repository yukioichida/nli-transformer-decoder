#!/usr/bin/env python3

import torch
from torchtext import data
from torchtext import datasets

from config import *
from log import logger


class SSTPreProcess:

    def __init__(self, device):
        self.device = device
        self.sentence_field = data.Field(include_lengths=True, fix_length=MAX_SEQ_SIZE, batch_first=True,
                                         eos_token="<eos>")
        self.label_field = data.LabelField(dtype=torch.int32, sequential=False)

        self.train_data, self.val_data, self.test_data = datasets.SST.splits(
            self.sentence_field, self.label_field, fine_grained=False, train_subtrees=True,
            filter_pred=lambda ex: ex.label != 'neutral')

    def build_iterators(self):
        self.sentence_field.build_vocab(self.train_data, max_size=MAX_VOCAB_SIZE)
        self.label_field.build_vocab(self.train_data)
        logger.info('Number of train/val/test dataset: {}/{}/{}'.format(len(self.train_data),
                                                                        len(self.val_data),
                                                                        len(self.test_data)))
        logger.info('Vocabulary size: {}'.format(len(self.sentence_field.vocab)))

        train_iter, valid_iter, test_iter = data.BucketIterator.splits((self.train_data, self.val_data, self.test_data),
                                                                       batch_size=BATCH_SIZE, device=self.device,
                                                                       repeat=False, shuffle=True)

        return train_iter, valid_iter, test_iter

    def include_positional_encoding(self, batch_matrix):
        """
        Include a new axis to inform whether the sequence represents the words or the positions
        :param batch_matrix: tensor[batch_size, sequence_length]
        :return: tensor [batch_size, sequence_length, (word or position index)]
        """
        # TODO: adjust matrix to variable length
        first_idx = len(self.sentence_field.vocab)
        last_idx = MAX_SEQ_SIZE + first_idx
        new_shape = (batch_matrix.size(0), batch_matrix.size(1), 2)
        formatted_batch = torch.zeros(new_shape, dtype=torch.int64, device=self.device)
        i = 0
        for element in batch_matrix:
            formatted_batch[i, :, 0] = element  # Word indexes
            formatted_batch[i, :, 1] = torch.arange(first_idx, last_idx, device=self.device)  # Positional indexes
            i += 1
        return formatted_batch
