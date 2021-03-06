#!/usr/bin/env python3

import torch

from torchtext import data
from torchtext import datasets

from modules.custom_dataset import SNLIBPEDataset, ContractDataset


class PreProcess:

    def __init__(self, device, logger, batch_size, base_path='.data'):
        self.device = device
        self.logger = logger
        self.batch_size = batch_size
        self.sentence_field = data.Field(include_lengths=True, batch_first=True, lower=True)
        self.label_field = data.LabelField()
        self.train_data, self.val_data, self.test_data = self.get_datasets(base_path=base_path)

    def build_vocab(self):
        self.sentence_field.build_vocab(self.train_data)
        self.label_field.build_vocab(self.train_data)

    def build_iterators(self, build_vocab=True):
        '''
            Create iterators for each dataset and builds the vocab.
            :param build_vocab - True whether vocab is not created and needed to be built
        :return: BucketIterator for each dataset
        '''
        if build_vocab:
            self.build_vocab()

        self.logger.info('Vocabulary size: {}'.format(len(self.sentence_field.vocab)))

        if self.val_data and self.test_data and self.train_data:
            # Train mode
            self.logger.info('Number of train/val/test dataset: {}/{}/{}'.format(len(self.train_data),
                                                                                 len(self.val_data),
                                                                                 len(self.test_data)))
            return data.BucketIterator.splits((self.train_data, self.val_data, self.test_data),
                                              batch_size=self.batch_size, device=self.device,
                                              repeat=False, shuffle=True)
        elif self.test_data:
            # Prediction mode
            self.logger.info('Prediction mode...')
            return data.BucketIterator.splits(self.test_data,
                                              batch_size=self.batch_size, device=self.device,
                                              repeat=False, shuffle=True)


class SNLIPreProcess(PreProcess):
    '''
        PreProcess class for Stanford Natural Language Inference dataset
    '''

    def get_datasets(self, base_path='.data'):
        return datasets.SNLI.splits(self.sentence_field, self.label_field, root=base_path)

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
        # max_premise_size = torch.max(premise[1]).item()
        # max_hyp_size = torch.max(hypothesis[1]).item()
        max_premise_size = premise[0].size(-1)
        max_hyp_size = hypothesis[0].size(-1)
        # prem_seq_sizes = premise[1].tolist()
        # hyp_seq_sizes = hypothesis[1].tolist()

        max_seq_len = max_premise_size + max_hyp_size + 1  # including separator and eos token
        # special token (end of sequence) that will contain all the prem-hyp information
        eos = len(self.sentence_field.vocab)
        # Positional encoding index regarding the relative position
        new_shape = (batch.batch_size, max_seq_len, 2)

        formatted_batch = torch.ones(new_shape, dtype=torch.int64, device=self.device)
        first_idx = eos + 1
        for idx in range(0, batch.batch_size):
            # [premise] + [hypothesis] + [eos token] TODO: test using another special token for prem-hyp separator
            # total_length = prem_seq_sizes[idx] + hyp_seq_sizes[idx] + 1
            total_length = premise[1][idx] + hypothesis[1][idx] + 1
            formatted_seq = torch.cat((premise[0][idx][:premise[1][idx]],
                                       hypothesis[0][idx][:hypothesis[1][idx]],
                                       torch.tensor([eos], device=self.device)))
            formatted_batch[idx, :total_length, 0] = formatted_seq  # Word indexes
            formatted_batch[idx, :total_length, 1] = torch.arange(first_idx, first_idx + total_length,
                                                                  device=self.device)  # Positional indexes
        return formatted_batch, batch.label.long()


class ContractPreProcess(PreProcess):

    def __init__(self, device, logger, batch_size, base_path='.data'):
        super().__init__(device, logger, batch_size, base_path)

    def load_pretrained_vocab(self, vocab):
        self.sentence_field.vocab = vocab

    def get_datasets(self, base_path='.data'):
        return None, None, ContractDataset.splits(self.sentence_field, self.label_field, root=base_path)

    def prepare_model_input(self, norm1, norm2, eos_index, device):
        """

        :param norm1: Normative sentence text
        :param norm2: Normative sentence text
        :param eos_index: index of end-of-sentence contained in vocabulary
        :param device: execution torch device
        :return: tensor with index concatenated
        """
        tensor_norm1, norm1_size = self.sentence_field.process([norm1.split()], device)
        tensor_norm2, norm2_size = self.sentence_field.process([norm2.split()], device)
        eos_tensor = torch.tensor([eos_index], device=device)

        max_seq_len = norm1_size.item() + norm2_size.item() + 1
        first_idx = eos_index + 1
        last_idx = max_seq_len + first_idx

        new_shape = (1, max_seq_len, 2)
        formatted_batch = torch.zeros(new_shape, dtype=torch.int64, device=self.device)

        formatted_batch[0, :, 0] = torch.cat([tensor_norm1[0], tensor_norm2[0], eos_tensor])
        formatted_batch[0, :, 1] = torch.arange(first_idx, last_idx, device=self.device)

        return formatted_batch
