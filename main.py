import os
from global_settings import LOG_PATH
from global_settings import CLEAN_PATH
from tools.log import init_data_log
from data_prep.data_clean import clean_data
from data_prep.data_clean import split_data

# Clean data
if __name__ == "__main__":
    if not os.path.isdir(LOG_PATH):
        os.mkdir(LOG_PATH)

    if not os.path.isdir(CLEAN_PATH):
        os.mkdir(CLEAN_PATH)

    raw_file = "raw.csv"
    date_file = "data.csv"
    clean_file = "cleaned.csv"

    init_data_log()
    clean_data(date_file, clean_file)
    # split_data(clean_file, num=250)
