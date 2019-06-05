#!/usr/bin/env python3

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage


class Trainer:

    def __init__(self, model, optimizer, loss_function, prepare_batch_fn):
        self.trainer = create_supervised_trainer(model, optimizer, loss_function, prepare_batch=prepare_batch_fn)
        self.evaluator = create_supervised_evaluator(model, metrics={"accuracy": Accuracy()})


    def train(self, train_iterator, val_iterator, test_iterator):