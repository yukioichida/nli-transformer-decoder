import torch
from torchtext import data
from torchtext import datasets
from torch.nn import CrossEntropyLoss
import random

"""
sentence_field = data.Field(include_lengths=True, batch_first=True, eos_token="<eos>")
label_field = data.LabelField(dtype=torch.int32, sequential=False)

# make splits for data
train, val, test = datasets.SST.splits(
    sentence_field, label_field, fine_grained=False, train_subtrees=True,
    filter_pred=lambda ex: ex.label != 'neutral')

# print information about the data
print('train.fields', train.fields)
print('len(train)', len(train))
print('vars(train[0])', vars(train[0]))

sentence_field.build_vocab(train)
label_field.build_vocab(train)

print(label_field.vocab.freqs)
print('Labels', len(label_field.vocab))

train_iter, valid_iter, test_iter = data.BucketIterator.splits((train, val, test),
                                                               batch_size=32, device='cpu',
                                                               repeat=False, shuffle=True)
sizes = []
for batch in train_iter:
    text = batch.text[0]
    seq_size = batch.text[1]
    max_size = torch.max(seq_size)
    sizes.append(max_size)
    label = batch.label

print(max(sizes))
print(len(train_iter))"""

output = torch.tensor([[77.4545, 2.234], [-1.23, 9.8]])
expected = torch.tensor([0, 0])
print(output.argmax(1))

loss = CrossEntropyLoss()
print("loss {:.4f}".format(loss(output, expected)))
