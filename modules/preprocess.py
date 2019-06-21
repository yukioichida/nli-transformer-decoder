#!/usr/bin/env python3

import torch

from torchtext import data
from torchtext import datasets

from modules.custom_dataset import SNLIBPEDataset

class PreProcess:

    def __init__(self, device, logger, max_prem_size, max_hyp_size, batch_size):
        self.device = device
        self.logger = logger
        self.max_prem_size = max_prem_size
        self.max_hyp_size = max_hyp_size
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
        max_premise_size = min(torch.max(premise[1]).item(), self.max_prem_size)
        max_hyp_size = min(torch.max(hypothesis[1]).item(), self.max_hyp_size)

        max_seq_len = max_premise_size + max_hyp_size + 1  # including separator and eos token
        # special token (end of sequence) that will contain all the prem-hyp information
        eos = len(self.sentence_field.vocab)
        # Positional encoding index regarding the relative position
        first_idx = eos + 1
        last_idx = max_seq_len + first_idx
        new_shape = (batch.batch_size, max_seq_len, 2)

        formatted_batch = torch.zeros(new_shape, dtype=torch.int64, device=self.device)
        for idx in range(0, batch.batch_size):
            # [premise] + [hypothesis] + [eos token] TODO: test using another special token for prem-hyp separator
            formatted_seq = torch.cat((premise[0][idx][:max_premise_size],
                                       hypothesis[0][idx][:max_hyp_size],
                                       torch.tensor([eos], device=self.device)))
            formatted_batch[idx, :, 0] = formatted_seq  # Word indexes
            formatted_batch[idx, :, 1] = torch.arange(first_idx, last_idx, device=self.device)  # Positional indexes
        return formatted_batch, batch.label.long()


class SNLIBPEPreProcess(SNLIPreProcess):
    '''
        Preprocess class for SNLI dataset bpe format
    '''

    def __init__(self, device, logger, max_prem_size, max_hyp_size, batch_size,
                 train_file='snli-train.20000.bpe.tsv', val_file='snli-val.20000.bpe.tsv',
                 test_file='snli-test.20000.bpe.tsv', base_path='.data/snli-bpe/'):
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.base_path = base_path
        super().__init__(device, logger, max_prem_size, max_hyp_size, batch_size)

    def get_datasets(self):
        return SNLIBPEDataset.splits(self.sentence_field, self.label_field)


class SSTPreProcess(PreProcess):
    """
    Preprocess class for SST dataset
    """

    def get_datasets(self):
        return datasets.SST.splits(self.sentence_field, self.label_field,
                                   fine_grained=False, train_subtrees=True,
                                   filter_pred=lambda
                                       ex: ex.label != 'neutral')

    def include_positional_encoding(self, batch, device, non_blocking=False):
        """
        Include a new axis to inform whether the sequence represents index of words or the positions.

        :param batch_matrix: tensor[batch_size, sequence_length]
        :return: tensor [batch_size, sequence_length, (word or position index)]
        """
        # TODO: adjust matrix to variable length

        x, y = batch.text[0], batch.label

        first_idx = len(self.sentence_field.vocab)
        last_idx = MAX_SEQ_SIZE + first_idx
        new_shape = (x.size(0), x.size(1), 2)
        formatted_batch = torch.zeros(new_shape, dtype=torch.int64, device=self.device)
        i = 0
        for element in x:
            formatted_batch[i, :, 0] = element  # Word indexes
            formatted_batch[i, :, 1] = torch.arange(first_idx, last_idx, device=self.device)  # Positional indexes
            i += 1
        return formatted_batch, y.long()
