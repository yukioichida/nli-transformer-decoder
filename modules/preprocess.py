#!/usr/bin/env python3

import torch

from torchtext import data
from torchtext import datasets

from modules.custom_dataset import SNLIBPEDataset


class PreProcess:

    def __init__(self, device, logger, batch_size):
        self.device = device
        self.logger = logger
        self.batch_size = batch_size
        self.sentence_field = data.Field(include_lengths=True, batch_first=True, lower=True)
        self.label_field = data.LabelField()
        self.train_data, self.val_data, self.test_data = self.get_datasets()

    def build_iterators(self):
        '''
            Create iterators for each dataset and builds the vocab.
        :return: BucketIterator for each dataset
        '''
        self.sentence_field.build_vocab(self.train_data)
        self.label_field.build_vocab(self.train_data)
        self.logger.info('Number of train/val/test dataset: {}/{}/{}'.format(len(self.train_data),
                                                                             len(self.val_data),
                                                                             len(self.test_data)))
        self.logger.info('Vocabulary size: {}'.format(len(self.sentence_field.vocab)))

        return data.BucketIterator.splits((self.train_data, self.val_data, self.test_data),
                                          batch_size=self.batch_size, device=self.device,
                                          repeat=False, shuffle=True)

    def get_datasets(self):
        return None, None, None


class SNLIPreProcess(PreProcess):
    '''
        PreProcess class for Stanford Natural Language Inference dataset
    '''

    def get_datasets(self):
        return datasets.SNLI.splits(self.sentence_field, self.label_field)

    def include_positional_encoding(self, batch, device, non_blocking=False):
        """
        Include a new axis to inform whether the sequence represents index of words or the positions.
        In the case of SNLI, we concatenate two sentence index vector
        :param batch_matrix: tensor[batch_size, sequence_length]
        :return: tensor [batch_size, sequence_length, (word or position index)]
        """

        premise = batch.premise
        hypothesis = batch.hypothesis
        # get the size of sentences to retrieve the longest sequence of batch
        max_premise_size = torch.max(premise[1]).item()
        max_hyp_size = torch.max(hypothesis[1]).item()
        prem_seq_sizes = premise[1].tolist()
        hyp_seq_sizes = hypothesis[1].tolist()

        max_seq_len = max_premise_size + max_hyp_size + 1  # including separator and eos token
        # special token (end of sequence) that will contain all the prem-hyp information
        eos = len(self.sentence_field.vocab)
        # Positional encoding index regarding the relative position
        new_shape = (batch.batch_size, max_seq_len, 2)

        formatted_batch = torch.ones(new_shape, dtype=torch.int64, device=self.device)
        first_idx = eos + 1
        for idx in range(0, batch.batch_size):
            # [premise] + [hypothesis] + [eos token] TODO: test using another special token for prem-hyp separator
            total_length = prem_seq_sizes[idx] + hyp_seq_sizes[idx] + 1
            formatted_seq = torch.cat((premise[0][idx][:prem_seq_sizes[idx]],
                                       hypothesis[0][idx][:hyp_seq_sizes[idx]],
                                       torch.tensor([eos], device=self.device)))
            formatted_batch[idx, :total_length, 0] = formatted_seq  # Word indexes
            formatted_batch[idx, :total_length, 1] = torch.arange(first_idx, first_idx+total_length, device=self.device)  # Positional indexes
        return formatted_batch, batch.label.long()
