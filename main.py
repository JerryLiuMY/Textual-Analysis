import os
from global_settings import DATA_FILE
from global_settings import LOG_PATH
from global_settings import CLEAN_PATH, CLEAN_FILE
from tools.log import init_data_log
from data_prep.data_clean import clean_data


# Clean data
if __name__ == "__main__":
    if not os.path.isdir(LOG_PATH):
        os.mkdir(LOG_PATH)

    if not os.path.isdir(CLEAN_PATH):
        os.mkdir(CLEAN_PATH)

    init_data_log()
    clean_data(DATA_FILE, CLEAN_FILE)
