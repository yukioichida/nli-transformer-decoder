import pandas as pd
import torch
from pandas import DataFrame

from modules.preprocess import SNLIPreProcess, ContractPreProcess
from modules.log import get_logger
from modules.model import TransformerDecoder
import os

print(os.path.dirname(os.path.dirname(__file__)))
print(os.getcwd())

BASE_PATH = ".data/contract-datasets/"
CONTRACT_DATASET_FILE = BASE_PATH + "all_contracts.tsv"

BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = get_logger('contract_analysis')

logger.info('loading norm dataset')
df_contract = pd.read_csv(CONTRACT_DATASET_FILE, sep='\t')

logger.info('loading vocabulary...')
preprocess = SNLIPreProcess(device, logger, 48, 28, BATCH_SIZE, base_path='.data')
preprocess.build_vocab()
train_vocab = preprocess.sentence_field.vocab

label_vocab = preprocess.label_field.vocab
class_vocab = []
for k, v in label_vocab.stoi.items():
    class_vocab.append('{}: {}'.format(k, v))
logger.info('class vocabulary: {}'.format('|'.join(class_vocab)))
logger.info('setup norm dataset')

contract_preprocess = ContractPreProcess(device, logger, 48, 28, BATCH_SIZE, base_path='.data')
contract_preprocess.load_pretrained_vocab(train_vocab)
contract_vocab = contract_preprocess.sentence_field.vocab

test_iter = contract_preprocess.build_iterators(build_vocab=False)

# Predicting NLI classes for norms
PRETRAINED_WEIGHTS = "saved_models/id-SNLI-12blk-12h-120d-8batch_model_52_acc=0.7919122.pth"

vocab_size = len(train_vocab)
max_seq_size = 48 + 28 + 1
eos_vocab_index = vocab_size
n_classes = len(preprocess.label_field.vocab)

norm1 = "this is insane"
norm2 = "although is hard, i will not surrender"
print("Tensor: " + str(tensor))

logger.info('loading model...')
model = TransformerDecoder(vocab_size=vocab_size, max_seq_length=max_seq_size,
                           word_embedding_dim=120, n_heads=12, n_blocks=12,
                           output_dim=n_classes, eos_token=eos_vocab_index)
model.load_state_dict(torch.load(PRETRAINED_WEIGHTS))
model = model.to(device)
model.eval()



# for each row in dataframe of all contracts
    # tensor = contract_preprocess.prepare_model_input(norm1, norm2, eos_index=eos_vocab_index, device=device)
    # predict = model(tensor)
    #index = torch.argmax(predict)
    #pred_class = label_vocab.itos[index.item()]
    # new_dataframe['norm1'] = norm1
    # new_dataframe['norm2'] = norm2
    # new_dataframe['conflict'] = conflict
    # new_dataframe['relation'] = pred_class

for index, row in df_contract.iterrows():
    norm1 = df_contract.iloc[index]['norm1']
    norm2 = df_contract.iloc[index]['norm2']
    tensor = contract_preprocess.prepare_model_input(norm1, norm2, eos_index=eos_vocab_index, device=device)
    predict = model(tensor)
    index = torch.argmax(predict)
    pred_class = label_vocab.itos[index.item()]
    df_contract.at[index, 'result'] = pred_class

df_contract.to_csv('.data/results/result.tsv', sep='\t')

#index = torch.argmax(predict)
#pred_class = label_vocab.itos[index.item()]

#logger.info("output tensor: {} - Value predicted: {} - Class predicted".format(predict, index, pred_class))
