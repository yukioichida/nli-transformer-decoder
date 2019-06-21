#!/usr/bin/env python3

from ignite.contrib.handlers import ProgressBar
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss


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
        self.device = device
        self.loss_function = loss_function
        self.use_progress_bar = use_progress_bar
        self.logger = logger
        self.prepare_batch_fn = prepare_batch_fn

    def train(self, train_iterator, val_iterator, test_iterator, epochs):
        if self.use_progress_bar:
            pbar = ProgressBar(persist=True, bar_format="")
            pbar.attach(self.trainer)

        @self.trainer.on(Events.STARTED)
        def start_callback(engine):
            engine.state.best_acc = 0

        @self.trainer.on(Events.EPOCH_COMPLETED, self.model)
        def log_training_results(engine, model):
            train_state = self.evaluator.run(train_iterator)
            val_state = self.evaluator.run(val_iterator)
            train_metrics = train_state.metrics
            val_metrics = val_state.metrics

            if engine.state.best_acc < val_metrics['accuracy']:
                engine.state.best_acc = val_metrics['accuracy']
                engine.state.best_model = model.state_dict()
                engine.state.best_epoch = engine.state.epoch

            message = "Epoch: {}  Train[acc: {:.4f}, loss: {:.4f}] - Val[acc: {:.4f}, loss: {:.4f}]" \
                .format(engine.state.epoch, train_metrics['accuracy'], train_metrics['loss'],
                        val_metrics['accuracy'], val_metrics['loss'])
            self.logger.info(message)

        @self.trainer.on(Events.COMPLETED)
        def log_test_results(engine):
            best_model = engine.state.best_model
            best_epoch = engine.state.best_epoch
            self.model.load_state_dict(best_model)
            test_evaluator = create_supervised_evaluator(self.model, metrics={"accuracy": Accuracy(),
                                                                              "loss": Loss(self.loss_function)},
                                                         device=self.device, prepare_batch=self.prepare_batch_fn)
            test_evaluator.run(test_iterator)
            self.log_output_summary(test_evaluator.state.metrics, best_epoch)

        self.trainer.run(train_iterator, max_epochs=epochs)

    def log_output_summary(self, metrics, best_epoch):
        message = """TEST SET RESULT - Using Epoch {} 
            - Avg Accuracy: {:.4f}
            - Avg Loss: {:.4f}
        """.format(best_epoch, metrics['accuracy'], metrics['loss'])
        self.logger.info(message)
