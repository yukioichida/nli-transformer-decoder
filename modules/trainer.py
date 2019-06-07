#!/usr/bin/env python3

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from modules.config import MAX_EPOCH


class Trainer:

    def __init__(self, model, optimizer, loss_function, prepare_batch_fn, device, logger, use_progress_bar=True):
        """

        :param model: Model for training
        :param optimizer: Optimizer algorithm used to train the model
        :param loss_function: Loss Function used to measure the model error
        :param prepare_batch_fn: Callback that adjust batch to be passed on model
        :param device: device used for training execution
        :param use_progress_bar: Whether use a progress bar to show the training status
        """
        self.trainer = create_supervised_trainer(model, optimizer, loss_function, device,
                                                 prepare_batch=prepare_batch_fn)
        self.evaluator = create_supervised_evaluator(model,
                                                     metrics={"accuracy": Accuracy(), "loss": Loss(loss_function)},
                                                     device=device, prepare_batch=prepare_batch_fn)
        self.model = model
        self.loss_function = loss_function
        self.use_progress_bar = use_progress_bar
        self.logger = logger

    def train(self, train_iterator, val_iterator, test_iterator):
        if self.use_progress_bar:
            pbar = ProgressBar(persist=True, bar_format="")
            pbar.attach(self.trainer)

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            self.evaluator.run(train_iterator)
            metrics = self.evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']
            self.logger.info("Training Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}"
                             .format(engine.state.epoch, avg_accuracy, avg_loss))

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            self.evaluator.run(val_iterator)
            metrics = self.evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']
            self.logger.info("Validation Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}"
                             .format(engine.state.epoch, avg_accuracy, avg_loss))

        @self.trainer.on(Events.COMPLETED)
        def log_test_results(engine):
            self.evaluator.run(test_iterator)
            metrics = self.evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['loss']
            self.logger.info("Validation Results - Epoch: {}  Avg accuracy: {:.4f} Avg loss: {:.4f}"
                             .format(engine.state.epoch, avg_accuracy, avg_loss))

        self.trainer.run(train_iterator, max_epochs=MAX_EPOCH)
