import os
import json
import numpy as np
import pandas as pd

# data folder
DATA_PATH = "/Users/mingyu/Desktop/data"
CLEAN_PATH = os.path.join(DATA_PATH, "cleaned")

# output folder
OUTPUT_PATH = "/Users/mingyu/Desktop/output"
LOG_PATH = os.path.join(OUTPUT_PATH, "log")
FIG_PATH = os.path.join(OUTPUT_PATH, "fig")

# stkcd
stkcd_all = list(np.load(os.path.join(DATA_PATH, "stkcd_all.npy")))
dalym = pd.read_csv(os.path.join("dalym.csv"))

# risklab server
user = "risklab_user"
host = "128.135.196.208"
PASS_PATH = os.path.join(DATA_PATH, "password")
with open(os.path.join(PASS_PATH, "password.json"), "r") as f:
    pass_file = json.load(f)
    password = pass_file["password"]
