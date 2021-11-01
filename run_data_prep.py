import os
from global_settings import RAW_FILE, DATA_FILE, CLEAN_FILE
from global_settings import CLEAN_PATH
from global_settings import LOG_PATH
# from data_prep.data_clean import save_data
from data_prep.data_clean import clean_data

# create directories
if not os.path.isdir(CLEAN_PATH):
    os.mkdir(CLEAN_PATH)

if not os.path.isdir(LOG_PATH):
    os.mkdir(LOG_PATH)


if __name__ == "__main__":
    from tools.log import init_data_log
    init_data_log()
    # save_data(RAW_FILE, DATA_FILE)
    clean_data(DATA_FILE, CLEAN_FILE)
