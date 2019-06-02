#!/usr/bin/env python3

import random
import torch
import torch.optim as optim
import torch.nn as nn
from torchtext import data
from torchtext import datasets
from config import *
from log import logger
from model import TransformerDecoder

from torch.nn.utils import clip_grad_norm_
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator

from ignite.metrics import Accuracy, Loss, RunningAverage

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

logger.info("Loading IMDB dataset...")

sentence_field = data.Field(include_lengths=True, fix_length=MAX_SEQ_SIZE, batch_first=True, eos_token="<eos>")
label_field = data.LabelField(dtype=torch.int32)

train_data, test_data = datasets.IMDB.splits(sentence_field, label_field)
train_data, valid_data = train_data.split(split_ratio=0.8, random_state=random.seed(SEED))

sentence_field.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
label_field.build_vocab(train_data)

sentence_field.build_vocab(train_data, max_size=MAX_VOCAB_SIZE)
label_field.build_vocab(train_data)

train_iter, valid_iter, test_iter = data.BucketIterator.splits((train_data, valid_data, test_data),
                                                               batch_size=BATCH_SIZE, device=device,
                                                               repeat=False, shuffle=True)

vocab_size = len(sentence_field.vocab)
output_size = len(label_field.vocab)

eos_vocab_index = sentence_field.vocab.stoi['<eos>']

#EOS_INDEX = vocab_size  # Index of END OF SENTENCE token in embedding matrix
POS_IDX_START = vocab_size + 1  # First index of position encoding in embedding matrix
POS_IDX_END = MAX_SEQ_SIZE + POS_IDX_START  # Last index of position encoding

logger.info('vocab size: {}'.format(vocab_size))
logger.info('Number of classes: {}'.format(output_size))
model = TransformerDecoder(vocab_size, MAX_SEQ_SIZE, WORD_DIM,
                           n_layers=QTTY_DECODER_BLOCK, n_heads=ATTENTION_HEADS,
                           output_dim=output_size, eos_token=eos_vocab_index)
model = model.to(device)

#learnable_params = filter(lambda param: param.requires_grad, model.parameters())
optimizer = optim.Adam(model.parameters())
loss_function = torch.nn.CrossEntropyLoss()
#loss_function = torch.nn.NLLLoss()
loss_function = loss_function.to(device)


def get_formatted_batch(batch_matrix, first_idx, last_idx):
    """
    Include a new axis to inform whether the sequence represents the words or the positions
    :param batch_matrix: tensor[batch_size, sequence_length]
    :param first_idx: Index of first positional encoding
    :param last_idx: Index of last positional encoding
    :return: tensor [batch_size, sequence_length, word or position index]
    """
    new_shape = (batch_matrix.size(0), batch_matrix.size(1), 2)
    formatted_batch = torch.zeros(new_shape, dtype=torch.int64, device=device)
    i = 0
    for element in batch_matrix:
        formatted_batch[i, :, 0] = element  # Word indexes
        formatted_batch[i, :, 1] = torch.arange(first_idx, last_idx, device=device)  # Positional indexes
        i += 1
    return formatted_batch

def predict(batch):
    x, y = batch.text[0], batch.label
    x = get_formatted_batch(x, POS_IDX_START, POS_IDX_END)
    return model(x), y

def process_function(engine, batch):
    '''
    Function that is executed for all processed batch
    '''
    model.train()
    optimizer.zero_grad()
    y_pred, y = predict(batch)
    loss = loss_function(y_pred, y.long())
    loss.backward()
    #clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def eval_function(engine, batch):
    model.eval()
    with torch.no_grad():
        y_pred, y = predict(batch)
        return y_pred, y.long()

def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = torch.round(y_pred)
    return y_pred, y


trainer = Engine(process_function)
train_evaluator = Engine(eval_function)
validator_evaluator = Engine(eval_function)

RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')
Accuracy(output_transform=thresholded_output_transform).attach(train_evaluator, 'accuracy')
Loss(loss_function).attach(train_evaluator, 'loss_train')  # cross entropy
Accuracy(output_transform=thresholded_output_transform).attach(validator_evaluator, 'accuracy')
Loss(loss_function).attach(validator_evaluator, 'loss_val')
pbar = ProgressBar(persist=True, bar_format="")
pbar.attach(trainer, ['loss'])


@trainer.on(Events.EPOCH_COMPLETED)
def log_training_results(engine):
    train_evaluator.run(train_iter)
    metrics = train_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_loss = metrics['loss_train']
    pbar.log_message(
        "Training Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}"
            .format(engine.state.epoch, avg_accuracy, avg_loss))

def log_validation_results(engine):
    validator_evaluator.run(valid_iter)
    metrics = validator_evaluator.state.metrics
    avg_accuracy = metrics['accuracy']
    avg_val_loss = metrics['loss_val']
    pbar.log_message(
        "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}"
            .format(engine.state.epoch, avg_accuracy, avg_val_loss))
    pbar.n = pbar.last_print_n = 0


trainer.add_event_handler(Events.EPOCH_COMPLETED, log_validation_results)

# TODO: model checkpoint
trainer.run(train_iter, max_epochs=MAX_EPOCH)


def evaluate(iterator):
    epoch_loss = 0

    model.eval()

    with torch.no_grad():
        for batch in iterator:
            predictions, y = predict(batch)
            loss = loss_function(predictions, y.long())
            epoch_loss += loss.item()
    print('epoch_loss {}'.format(epoch_loss))
    print('number of elements: {}'.format(len(iterator)))
    return epoch_loss / len(iterator)


test_loss = evaluate(test_iter)

print("Test Loss: %04fgit d" % test_loss)
