#!/usr/bin/env python3

from log import logger
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from config import MAX_EPOCH


class Trainer:

    def __init__(self, model, optimizer, loss_function, prepare_batch_fn, device, use_progress_bar=True):
        self.trainer = create_supervised_trainer(model, optimizer, loss_function, device,
                                                 prepare_batch=prepare_batch_fn)
        self.evaluator = create_supervised_evaluator(model,
                                                     metrics={"accuracy": Accuracy(), "loss": Loss(loss_function)},
                                                     device=device, prepare_batch=prepare_batch_fn)
        self.model = model
        self.loss_function = loss_function
        self.use_progress_bar = use_progress_bar

    def train(self, train_iterator, val_iterator, test_iterator):
        logger.info("Start train")
        if self.use_progress_bar:
            pbar = ProgressBar(persist=True, bar_format="")
            pbar.attach(self.trainer)

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            self.evaluator.run(train_iterator)
            metrics = self.evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']
            logger.info("Training Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}"
                        .format(engine.state.epoch, avg_accuracy, avg_loss))

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            self.evaluator.run(val_iterator)
            metrics = self.evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']
            logger.info("Validation Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}"
                        .format(engine.state.epoch, avg_accuracy, avg_loss))

        self.trainer.run(train_iterator, max_epochs=MAX_EPOCH)
