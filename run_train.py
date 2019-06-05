#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage

from config import *
from custom_optimizers import OpenAIAdam
from log import logger
from model import TransformerDecoder
from preprocess import SSTPreProcess

SEED = 1234
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

preprocess = SSTPreProcess(device)

train_iter, val_iter, test_iter = preprocess.build_iterators()

num

model = TransformerDecoder(vocab_size, MAX_SEQ_SIZE, WORD_DIM,
                           n_layers=QTTY_DECODER_BLOCK, n_heads=ATTENTION_HEADS, dropout=DROPOUT,
                           output_dim=output_size, eos_token=eos_vocab_index)
model = model.to(device)

nr_optimizer_update = (train_size // BATCH_SIZE) * MAX_EPOCH
optimizer = OpenAIAdam(model.parameters(), nr_optimizer_update)
loss_function = torch.nn.CrossEntropyLoss()
loss_function = loss_function.to(device)


def predict(batch):
    x, y = batch.text[0], batch.label
    x = get_formatted_batch(x, POS_IDX_START, POS_IDX_END)
    y_pred = model(x)
    return y_pred, y


def process_function(engine, batch):
    '''
    Function that is executed for all processed batch
    '''
    model.train()
    optimizer.zero_grad()
    y_pred, y = predict(batch)
    loss = F.cross_entropy(y_pred, y.long())
    loss.backward()
    # clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


def eval_function(engine, batch):
    with torch.no_grad():
        model.eval()
        y_pred, y = predict(batch)
        return y_pred, y.long()


def thresholded_output_transform(output):
    y_pred, y = output
    y_pred = y_pred.argmax(1)
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
    message = "Validation Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}" \
        .format(engine.state.epoch, avg_accuracy, avg_val_loss)
    pbar.log_message(message)
    logger.info(message)
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

print("Test Loss: %04f" % test_loss)
