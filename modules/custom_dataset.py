#!/usr/bin/env python3

import torchtext.data as data


class SNLIBPEDataset(data.TabularDataset):

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(
            len(ex.premise), len(ex.hypothesis))

    @classmethod
    def splits(cls, text_field, label_field, parse_field=None,
               extra_fields={}, root='.data', train='snli-train.20000.bpe.tsv',
               validation='snli-val.20000.bpe.tsv', test='snli-test.20000.bpe.tsv'):
        """Create dataset objects for splits of the SNLI dataset BPE FORMAT.

        Arguments:
            text_field: The field that will be used for premise and hypothesis
                data.
            label_field: The field that will be used for label data.
            parse_field: The field that will be used for shift-reduce parser
                transitions, or None to not include them.
            extra_fields: A dict[json_key: Tuple(field_name, Field)]
            root: The root directory that contained all datasets
            train: The filename of the train data.
            validation: The filename of the validation data, or None to not
                load the validation set.
            test: The filename of the test data, or None to not load the test
                set.
        """
        fields = {'premise': ('premise', text_field),
                  'hypothesis': ('hypothesis', text_field),
                  'label': ('label', label_field)}
        path = '.data/snli-bpe'
        return super(SNLIBPEDataset, cls).splits(
            path, root, train, validation, test,
            format='tsv', fields=fields)
