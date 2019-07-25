#!/usr/bin/env python3
import pandas as pd
import torch
from torch.nn import Softmax
from tqdm import tqdm

from modules.log import get_logger
from modules.model import TransformerDecoder
from modules.preprocess import SNLIPreProcess, ContractPreProcess

BASE_PATH = ".data/contract-datasets/"
CONTRACT_DATASET_FILE = BASE_PATH + "all_contracts.tsv"
BATCH_SIZE = 32
MAX_SEQ_SIZE = 360
# Predicting NLI classes for norms
PRETRAINED_WEIGHTS = "saved_models/SNLI-12blk-12h-240d-16batch_model_12_acc=0.8134526.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger('contract_analysis')
logger.info('loading norm dataset')
df_contract = pd.read_csv(CONTRACT_DATASET_FILE, sep='\t')

logger.info('loading vocabulary...')
preprocess = SNLIPreProcess(device, logger, BATCH_SIZE)
preprocess.build_vocab()
train_vocab = preprocess.sentence_field.vocab

label_vocab = preprocess.label_field.vocab
class_vocab = []
for k, v in label_vocab.stoi.items():
    class_vocab.append('{}: {}'.format(k, v))
logger.info('class vocabulary: {}'.format('|'.join(class_vocab)))
logger.info('setup norm dataset')
contract_preprocess = ContractPreProcess(device, logger, BATCH_SIZE)
contract_preprocess.load_pretrained_vocab(train_vocab)
contract_vocab = contract_preprocess.sentence_field.vocab
test_iter = contract_preprocess.build_iterators(build_vocab=False)

vocab_size = len(train_vocab)
eos_vocab_index = vocab_size
n_classes = len(preprocess.label_field.vocab)

logger.info('loading model...')
model = TransformerDecoder(vocab_size=vocab_size, max_seq_length=MAX_SEQ_SIZE, word_embedding_dim=240, n_heads=12,
                           n_blocks=12, output_dim=n_classes, eos_token=eos_vocab_index)
model.load_state_dict(torch.load(PRETRAINED_WEIGHTS))
model = model.to(device)
model.eval()
softmax = Softmax(dim=-1)


def predict_on_norms(premise, hypothesis):
    tensor = contract_preprocess.prepare_model_input(premise[:179], hypothesis[:179], eos_index=eos_vocab_index,
                                                     device=device)
    predict = model(tensor)
    predict = softmax(predict)
    index = torch.argmax(predict).tolist()
    return label_vocab.itos[index], predict.tolist()[0]


def write_results():
    entailment_index = label_vocab.stoi['entailment']
    contradiction_index = label_vocab.stoi['contradiction']
    neutral_index = label_vocab.stoi['neutral']
    for index, row in tqdm(df_contract.iterrows()):
        norm1 = df_contract.iloc[index]['norm1']
        norm2 = df_contract.iloc[index]['norm2']
        majority_class, probabilities = predict_on_norms(norm1, norm2)
        df_contract.at[index, 'majority_class'] = majority_class
        df_contract.at[index, 'entailment_prob'] = probabilities[entailment_index]
        df_contract.at[index, 'contradiction_prob'] = probabilities[contradiction_index]
        df_contract.at[index, 'neutral_prob'] = probabilities[neutral_index]
        # Reverse mode: norm2->norm1
        rev_majority_class, rev_probabilities = predict_on_norms(norm2, norm1)
        df_contract.at[index, 'rev_majority_class'] = rev_majority_class
        df_contract.at[index, 'rev_entailment_prob'] = rev_probabilities[entailment_index]
        df_contract.at[index, 'rev_contradiction_prob'] = rev_probabilities[contradiction_index]
        df_contract.at[index, 'rev_neutral_prob'] = rev_probabilities[neutral_index]
        df_contract.at[index, 'conf_type'] = df_contract.iloc[index]['conf_type']
    df_contract.to_csv('result.tsv', sep='\t', index=False)


write_results()
