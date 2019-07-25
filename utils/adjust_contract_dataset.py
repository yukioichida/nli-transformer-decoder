#!/usr/bin/env python3
import pandas as pd

BASE_PATH = "../.data/contract-datasets/"
CONTRACT_CONFLICT_FILE = BASE_PATH + "conflicts.csv"
CONTRACT_NONCONFLICT_FILE = BASE_PATH + "non_conflicts.csv"
ALL_CONTRACT_FILE = BASE_PATH + "all_contracts.tsv"
NON_CONFLICT_TYPE = 5

APPEND_NONCONFLICTS = False

df_conflicts = pd.read_csv(CONTRACT_CONFLICT_FILE)
df_nonconflicts = pd.read_csv(CONTRACT_NONCONFLICT_FILE)

df_conflicts = df_conflicts[['norm1', 'norm2', 'conf_type']]
df_conflicts['conflict'] = True

if APPEND_NONCONFLICTS:
    df_nonconflicts = df_nonconflicts[['norm1', 'norm2']]
    df_nonconflicts['conflict'] = False
    df_nonconflicts['conf_type'] = NON_CONFLICT_TYPE
    df_contracts = df_conflicts.append([df_nonconflicts], sort=True)
else:
    df_contracts = df_conflicts

df_contracts.to_csv(ALL_CONTRACT_FILE, index=False, sep='\t')
