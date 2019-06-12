from modules.preprocess import SNLIPreProcess
from modules.log import get_logger
import torch

device = torch.device('cpu')
logger = get_logger("test")
preprocess = SNLIPreProcess(device, logger)
train_iter, val_iter, test_iter = preprocess.build_iterators()
print(len(preprocess.label_field.vocab))
print(preprocess.label_field.vocab.freqs)