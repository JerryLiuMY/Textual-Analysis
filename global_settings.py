import os
import json
DATA_PATH = "/Users/mingyu/Desktop/data"
RAW_FILE = "raw.csv"
DATA_FILE = "data.csv"

CLEAN_PATH = os.path.join(DATA_PATH, "clean")
CLEAN_FILE = "clean.csv"

LOG_PATH = os.path.join(DATA_PATH, "log")

# risklab server
user = "risklab_user"
host = "128.135.196.208"
with open(os.path.join(DATA_PATH, "password.json"), "r") as f:
    pass_file = json.load(f)
    password = pass_file["password"]
