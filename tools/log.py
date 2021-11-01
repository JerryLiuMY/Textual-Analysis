import json
import os
from global_settings import LOG_PATH


def init_data_log():
    data_log = {
        "original": 0,
        "available": 0,
        "drop_nan": 0,
        "single_tag": 0,
        "match_stkcd": 0
    }

    with open(os.path.join(LOG_PATH, "data_log.json"), "w") as f:
        json.dump(data_log, f)

    print("data_log.json initialized")
