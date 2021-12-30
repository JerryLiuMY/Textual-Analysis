import os
import json
import numpy as np
import pandas as pd
from pathlib import Path

# directories
DESKTOP_PATH = str(Path(os.getcwd()).parent.absolute())
DATA_PATH = os.path.join(DESKTOP_PATH, "data")
OUTPUT_PATH = os.path.join(DESKTOP_PATH, "output")
CLEAN_PATH = os.path.join(DATA_PATH, "cleaned")
RICH_PATH = os.path.join(DATA_PATH, "enriched")
LOG_PATH = os.path.join(OUTPUT_PATH, "log")

# stkcd & trddt
stkcd_all = list(np.load(os.path.join(DATA_PATH, "stkcd_all.npy")))
dalym = pd.read_csv(os.path.join(DATA_PATH, "dalym.csv"))
trddt_all = np.array(sorted(set(dalym["Trddt"])))

# risklab server
user = "risklab_user"
host = "128.135.196.208"
with open(os.path.join(DATA_PATH, "password.json"), "r") as f:
    pass_file = json.load(f)
    password = pass_file["password"]

# dictionary
# https://github.com/MengLingchao/Chinese_financial_sentiment_dictionary
xlsx_dict = pd.ExcelFile(os.path.join(DATA_PATH, "Chinese_Dict.xlsx"))
pos_dict = [_.strip() for _ in xlsx_dict.parse("positive").iloc[:, 0]]
neg_dict = [_.strip() for _ in xlsx_dict.parse("negative").iloc[:, 0]]
full_dict = pos_dict + neg_dict
stop_list = list(pd.read_csv(os.path.join(DATA_PATH, "stop_list.txt"), header=None).iloc[:, 0])


# BERT and doc2vec
# sbatch script.sbatch
# squeue --user=mingyuliu
# scancel
