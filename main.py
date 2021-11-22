import os
import glob
from global_settings import LOG_PATH
from global_settings import CLEAN_PATH
from global_settings import RICH_PATH
from global_settings import WORD_PATH
from data_prep.data_clean import save_data
from data_prep.data_split import split_data
from data_prep.data_clean import clean_data
from data_prep.data_enrich import enrich_data
from data_prep.word_matrix import build_word
from multiprocessing.pool import Pool


# Create directories
if not os.path.isdir(LOG_PATH):
    os.mkdir(LOG_PATH)

if not os.path.isdir(CLEAN_PATH):
    os.mkdir(CLEAN_PATH)

if not os.path.isdir(RICH_PATH):
    os.mkdir(RICH_PATH)

if not os.path.isdir(WORD_PATH):
    os.mkdir(WORD_PATH)


if __name__ == "__main__":
    # # clean & split data
    # raw_file = "raw.csv"
    # data_file = "data.csv"
    # clean_file = "cleaned.csv"
    # split_num = 250
    # save_data(raw_file, data_file)
    # clean_data(data_file, clean_file)
    # split_data(clean_file, split_num=split_num)
    #
    # # enrich data
    # sub_file_clean_li = [_.split("/")[-1] for _ in glob.glob(os.path.join(CLEAN_PATH, "*"))]
    # sub_file_rich_idx = [_.split("/")[-1].split(".")[0].split("_")[1] for _ in glob.glob(os.path.join(RICH_PATH, "*"))]
    # sub_file_clean_li = sorted([_ for _ in sub_file_clean_li if _.split(".")[0].split("_")[1] not in sub_file_rich_idx])
    # num_proc = 8
    #
    # for idx in range(0, len(sub_file_clean_li), num_proc):
    #     pool = Pool(num_proc)
    #     pool.map(enrich_data, sub_file_clean_li[idx: idx + num_proc])
    #     pool.close()
    #     pool.join()

    # build word matrix
    sub_file_rich_li = [_.split("/")[-1] for _ in glob.glob(os.path.join(RICH_PATH, "*"))]
    sub_word_file_idx = [_.split("/")[-1].split(".")[0].split("_")[1] for _ in glob.glob(os.path.join(WORD_PATH, "*"))]
    sub_file_rich_li = sorted([_ for _ in sub_file_rich_li if _.split(".")[0].split("_")[1] not in sub_word_file_idx])
    num_proc = 8

    for idx in range(0, len(sub_file_rich_li), num_proc):
        pool = Pool(num_proc)
        pool.map(build_word, sub_file_rich_li[idx: idx + num_proc])
        pool.close()
        pool.join()
