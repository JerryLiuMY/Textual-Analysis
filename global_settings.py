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
date0_min = "2015-01-07"
date0_max = "2019-07-30"

# risklab server
user = "risklab_user"
host = "128.135.196.208"
with open("password.json", "r") as f:
    pass_file = json.load(f)
    password = pass_file["password"]

# dictionary
# https://github.com/MengLingchao/Chinese_financial_sentiment_dictionary
xlsx_dict = pd.ExcelFile(os.path.join(DATA_PATH, "Chinese_Dict.xlsx"))
pos_dict = [_.strip() for _ in xlsx_dict.parse("positive").iloc[:, 0]]
neg_dict = [_.strip() for _ in xlsx_dict.parse("negative").iloc[:, 0]]
full_dict = pos_dict + neg_dict
stop_list = list(pd.read_csv(os.path.join(DATA_PATH, "stop_list.txt"), header=None).iloc[:, 0])


# BERT
# solve issue of iterator=
# get_latest_training_loss
# speed up inference


# sinteractive --partition=broadwl-lc --nodes=1 --ntasks-per-node=8 --mem-per-cpu=4000 --time=36:00:00
# python3 main.py -m doc2vec -f 0 -t 5
# python3 main.py -m doc2vec -f 5 -t 10
# python3 main.py -m doc2vec -f 10 -t 15
# python3 main.py -m doc2vec -f 15 -t 20
# python3 main.py -m doc2vec -f 20 -t 25
# python3 main.py -m doc2vec -f 25 -t 30
# python3 main.py -m doc2vec -f 30 -t 35
# python3 main.py -m doc2vec -f 35 -t 37
