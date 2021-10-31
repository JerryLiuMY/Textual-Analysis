import os
import json
DATA_PATH = "/Users/mingyu/Desktop/data"
LOG_PATH = os.path.join(DATA_PATH, "log")

if not os.path.isdir(LOG_PATH):
    os.mkdir(LOG_PATH)

user = "risklab_user"
host = "128.135.196.208"
with open(os.path.join(DATA_PATH, "password.json"), "r") as f:
    pass_file = json.load(f)
    password = pass_file["password"]
