import os
import json
import numpy as np

# data folder
DATA_PATH = "/Users/mingyu/Desktop/data"
CLEAN_PATH = os.path.join(DATA_PATH, "clean")
RAW_FILE = "raw.csv"
DATA_FILE = "data.csv"
CLEAN_FILE = "clean.csv"

# output folder
OUTPUT_PATH = "/Users/mingyu/Desktop/output"
LOG_PATH = os.path.join(OUTPUT_PATH, "log")
FIG_PATH = os.path.join(OUTPUT_PATH, "fig")

# risklab server
user = "risklab_user"
host = "128.135.196.208"
with open(os.path.join(DATA_PATH, "password.json"), "r") as f:
    pass_file = json.load(f)
    password = pass_file["password"]

# stkcd
stkcd_all = list(np.load(os.path.join(DATA_PATH, "stkcd_all.npy")))