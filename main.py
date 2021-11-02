import os
import glob
from global_settings import LOG_PATH
from global_settings import CLEAN_PATH
from global_settings import RICH_PATH
from data_prep.data_clean import clean_data
from data_prep.data_clean import split_data
from data_prep.data_enrich import enrich_data

if not os.path.isdir(LOG_PATH):
    os.mkdir(LOG_PATH)

if not os.path.isdir(CLEAN_PATH):
    os.mkdir(CLEAN_PATH)

if not os.path.isdir(RICH_PATH):
    os.mkdir(RICH_PATH)

# Clean data
if __name__ == "__main__":
    raw_file = "raw.csv"
    data_file = "data.csv"
    clean_file = "cleaned.csv"
    split_num = 250

    # split data
    # save_data(raw_file, data_file)
    # clean_data(data_file, clean_file)
    # split_data(clean_file, split_num=split_num)

    # enrich data
    sub_file_li = sorted([_.split("/")[-1] for _ in glob.glob(os.path.join(CLEAN_PATH, "*"))])
    for sub_file in sub_file_li:
        enrich_data(sub_file)
