import pandas as pd
import torch
from pandas import DataFrame

from modules.preprocess import SNLIPreProcess, ContractPreProcess
from modules.log import get_logger
from modules.model import TransformerDecoder

BASE_PATH = ".data/contract-datasets/"
CONTRACT_DATASET_FILE = BASE_PATH + "all_contracts.tsv"

df_contract: DataFrame = pd.read_csv(CONTRACT_DATASET_FILE, sep='\t')

BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger('contract_analysis', '.')

preprocess = SNLIPreProcess(device, logger, 48, 28, BATCH_SIZE, base_path='.data')
preprocess.build_vocab()
train_vocab = preprocess.sentence_field.vocab

contract_preprocess = ContractPreProcess(device, logger, 48, 28, BATCH_SIZE, base_path='.data')
contract_preprocess.load_pretrained_vocab(train_vocab)
contract_vocab = contract_preprocess.sentence_field.vocab

test_iter = contract_preprocess.build_iterators(build_vocab=False)

# Predicting NLI classes for norms

PRETRAINED_WEIGHTS = "../saved_models/id-SNLI-12blk-12h-120d-8batch_model_52_acc=0.7919122.pth"

vocab_size = len(train_vocab)
max_seq_size = 48 + 28 + 1
eos_vocab_index = vocab_size
n_classes = len(preprocess.label_field.vocab)


norm1 = "this is insane"
norm2 = "although is hard, i will not surrender"
tensor = contract_preprocess.prepare_model_input(norm1, norm2, eos_index=eos_vocab_index, device=device)
print("Tensor: " + str(tensor))

model = TransformerDecoder(vocab_size=vocab_size, max_seq_length=max_seq_size,
                           word_embedding_dim=120, n_heads=12, n_blocks=12,
                           output_dim=n_classes, eos_token=eos_vocab_index)
model.load_state_dict(torch.load(PRETRAINED_WEIGHTS))

model.eval()

predict = model(tensor)
logger.info("Value predicted: {}".format(predict))
