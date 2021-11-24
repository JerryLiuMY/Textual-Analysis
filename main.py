import os
import glob
import pandas as pd
from scipy.sparse import load_npz
from global_settings import CLEAN_PATH, full_dict
from global_settings import RICH_PATH
from global_settings import WORD_PATH
from data_prep.data_clean import save_data
from data_prep.data_split import split_data
from data_prep.data_clean import clean_data
from data_prep.data_enrich import enrich_data
from data_proc.word_mtx import build_word_mtx
from multiprocessing.pool import Pool


def run_data_prep(raw_file="raw.csv", data_file="data.csv", clean_file="cleaned.csv"):
    """ Run data processing -- clean, split & enrich data
    :param raw_file: raw file
    :param data_file: data file
    :param clean_file: cleaned file
    :return:
    """
    # clean & split data
    split_num = 750
    save_data(raw_file, data_file)
    clean_data(data_file, clean_file)
    split_data(clean_file, split_num=split_num)

    # enrich data
    sub_file_clean_li = [_.split("/")[-1] for _ in glob.glob(os.path.join(CLEAN_PATH, "*"))]
    sub_file_rich_idx = [_.split("/")[-1].split(".")[0].split("_")[1] for _ in glob.glob(os.path.join(RICH_PATH, "*"))]
    sub_file_clean_li = sorted([_ for _ in sub_file_clean_li if _.split(".")[0].split("_")[1] not in sub_file_rich_idx])

    num_proc = 16
    for idx in range(0, len(sub_file_clean_li), num_proc):
        pool = Pool(num_proc)
        pool.imap(enrich_data, sub_file_clean_li[idx: idx + num_proc])
        pool.close()
        pool.join()


def run_word_mtx():
    """ Build word matrix"""
    # build word matrix
    sub_file_rich_li = [_.split("/")[-1] for _ in glob.glob(os.path.join(RICH_PATH, "*"))]
    sub_word_file_idx = [_.split("/")[-1].split(".")[0].split("_")[1] for _ in glob.glob(os.path.join(WORD_PATH, "*"))]
    sub_file_rich_li = sorted([_ for _ in sub_file_rich_li if _.split(".")[0].split("_")[1] not in sub_word_file_idx])

    num_proc = 16
    for idx in range(0, len(sub_file_rich_li), num_proc):
        pool = Pool(num_proc)
        pool.imap(build_word_mtx, sub_file_rich_li[idx: idx + num_proc])
        pool.close()
        pool.join()


def run_ssestm():
    """ Run ssestm"""
    # define file index
    sub_file_rich_idx = [_.split("/")[-1].split(".")[0].split("_")[1] for _ in glob.glob(os.path.join(RICH_PATH, "*"))]
    sub_word_file_idx = [_.split("/")[-1].split(".")[0].split("_")[1] for _ in glob.glob(os.path.join(WORD_PATH, "*"))]
    if sorted(sub_file_rich_idx) != sorted(sub_word_file_idx):
        raise ValueError("Mismatch between enriched files and word matrix files")

    sub_file_rich_li = sorted([_.split("/")[-1] for _ in glob.glob(os.path.join(RICH_PATH, "*"))])
    sub_word_file_li = sorted([_.split("/")[-1] for _ in glob.glob(os.path.join(WORD_PATH, "*"))])

    df_rich = pd.DataFrame()
    word_df = pd.DataFrame()

    for sub_file_rich, sub_word_file in zip(sub_file_rich_li, sub_word_file_li):
        sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich))
        sub_word_df = pd.DataFrame(load_npz(os.path.join(WORD_PATH, sub_word_file)).todense(), columns=full_dict)
        df_rich = df_rich.append(sub_df_rich)
        word_df = word_df.append(sub_word_df)
