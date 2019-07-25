#!/usr/bin/env python3

from ignite.contrib.handlers import ProgressBar
from ignite.handlers import EarlyStopping, ModelCheckpoint
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss


class Trainer:

    def __init__(self, model, optimizer, loss_function, prepare_batch_fn, device, logger, model_id,
                 use_progress_bar=True):
        """

        :param model: Model for training
        :param optimizer: Optimizer algorithm used to train the model
        :param loss_function: Loss Function used to measure the model error
        :param prepare_batch_fn: Callback that adjust batch to be passed on model
        :param device: device used for training execution
        :param model_id: identifier of the trained model
        :param use_progress_bar: Whether use a progress bar to show the training status
        """
        self.trainer = create_supervised_trainer(model, optimizer, loss_function, device,
                                                 prepare_batch=prepare_batch_fn)

        self.train_evaluator = create_supervised_evaluator(model,
                                                           metrics={"accuracy": Accuracy(),
                                                                    "loss": Loss(loss_function)},
                                                           device=device, prepare_batch=prepare_batch_fn)
        self.val_evaluator = create_supervised_evaluator(model,
                                                         metrics={"accuracy": Accuracy(),
                                                                  "loss": Loss(loss_function)},
                                                         device=device, prepare_batch=prepare_batch_fn)

        self.model = model
        self.device = device
        self.loss_function = loss_function
        self.use_progress_bar = use_progress_bar
        self.logger = logger
        self.prepare_batch_fn = prepare_batch_fn
        self.model_id = model_id

    def train(self, train_iterator, val_iterator, test_iterator, epochs):
        if self.use_progress_bar:
            pbar = ProgressBar(persist=True, bar_format="")
            pbar.attach(self.trainer)

        def score_function(engine):
            return engine.state.metrics['accuracy']

        @self.trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(engine):
            train_state = self.train_evaluator.run(train_iterator)
            val_state = self.val_evaluator.run(val_iterator)
            train_metrics = train_state.metrics
            val_metrics = val_state.metrics

            message = "Epoch: {}  Train[acc: {:.4f}, loss: {:.4f}] - Val[acc: {:.4f}, loss: {:.4f}]" \
                .format(engine.state.epoch, train_metrics['accuracy'], train_metrics['loss'],
                        val_metrics['accuracy'], val_metrics['loss'])
            self.logger.info(message)

        @self.trainer.on(Events.COMPLETED)
        def log_test_results(engine):
            test_metrics = self.val_evaluator.run(test_iterator)
            self.log_output_summary(test_metrics)

        checkpoint = ModelCheckpoint(dirname='saved_models/', filename_prefix=self.model_id,
                                     score_function=score_function,
                                     score_name='acc', n_saved=4, create_dir=True, save_as_state_dict=True)
        early_stop = EarlyStopping(patience=10, score_function=score_function, trainer=self.trainer)
        self.val_evaluator.add_event_handler(Events.COMPLETED, early_stop)
        self.val_evaluator.add_event_handler(Events.EPOCH_COMPLETED, checkpoint, {'model': self.model})
        self.trainer.run(train_iterator, max_epochs=epochs)

    def log_output_summary(self, metrics):
        message = """TEST SET RESULT 
            - Avg Accuracy: {:.4f}
            - Avg Loss: {:.4f}
        """.format(metrics['accuracy'], metrics['loss'])
        self.logger.info(message)
