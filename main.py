import os
from global_settings import LOG_PATH
from global_settings import CLEAN_PATH
from data_prep.data_clean import clean_data
from data_prep.data_clean import split_data

if not os.path.isdir(LOG_PATH):
    os.mkdir(LOG_PATH)

if not os.path.isdir(CLEAN_PATH):
    os.mkdir(CLEAN_PATH)

# Clean data
if __name__ == "__main__":
    raw_file = "raw.csv"
    data_file = "data.csv"
    clean_file = "cleaned.csv"

    # save_data(raw_file, data_file)
    clean_data(data_file, clean_file)
    split_data(clean_file, num=250)
