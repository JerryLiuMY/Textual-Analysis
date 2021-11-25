import os
import glob
import pandas as pd
import numpy as np
import scipy as sp
import datetime
from multiprocessing.pool import Pool
from scipy.sparse import load_npz, csr_matrix
from global_settings import CLEAN_PATH, full_dict
from global_settings import RICH_PATH
from global_settings import WORD_PATH
from data_prep.data_clean import save_data
from data_prep.data_clean import clean_data
from data_prep.data_split import split_data
from data_prep.data_enrich import enrich_data
from data_proc.word_sps import build_word_sps
from experiments.params import window_dict
from experiments.params import date0_min, date0_max
from experiments.generators import generate_window
from experiments.experiment import experiment


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
        pool.map(enrich_data, sub_file_clean_li[idx: idx + num_proc])
        pool.close()
        pool.join()


def run_word_sps():
    """
    Build word sparse matrix
    """

    sub_file_rich_li = [_.split("/")[-1] for _ in glob.glob(os.path.join(RICH_PATH, "*"))]
    sub_word_file_idx = [_.split("/")[-1].split(".")[0].split("_")[1] for _ in glob.glob(os.path.join(WORD_PATH, "*"))]
    sub_file_rich_li = sorted([_ for _ in sub_file_rich_li if _.split(".")[0].split("_")[1] not in sub_word_file_idx])

    num_proc = 12
    for idx in range(0, len(sub_file_rich_li), num_proc):
        pool = Pool(num_proc)
        pool.map(build_word_sps, sub_file_rich_li[idx: idx + num_proc])
        pool.close()
        pool.join()


def run_experiment(model_name, perc_ls):
    """ Run experiment
    :param model_name: model name
    :param perc_ls: percentage of long-short portfolio
    :return: ret_e, ret_v
    """

    # define index
    sub_file_rich_idx = [_.split("/")[-1].split(".")[0].split("_")[1] for _ in glob.glob(os.path.join(RICH_PATH, "*"))]
    sub_word_file_idx = [_.split("/")[-1].split(".")[0].split("_")[1] for _ in glob.glob(os.path.join(WORD_PATH, "*"))]
    if sorted(sub_file_rich_idx) != sorted(sub_word_file_idx):
        raise ValueError("Mismatch between enriched files and word matrix files")

    sub_file_rich_li = sorted([_.split("/")[-1] for _ in glob.glob(os.path.join(RICH_PATH, "*"))])
    sub_word_file_li = sorted([_.split("/")[-1] for _ in glob.glob(os.path.join(WORD_PATH, "*"))])

    # get df_rich & textual
    df_rich = pd.DataFrame()
    textual = csr_matrix(np.empty((0, len(full_dict)), np.int64))
    for sub_file_rich, sub_word_file in zip(sub_file_rich_li, sub_word_file_li):
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Combining {sub_file_rich} and {sub_word_file}")
        sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich))
        sub_word_sps = load_npz(os.path.join(WORD_PATH, sub_word_file))
        df_rich = df_rich.append(sub_df_rich)
        textual = sp.sparse.vstack([textual, sub_word_sps], format="csr")

    df_rich.reset_index(inplace=True, drop=True)

    # rolling window iterator
    window_iter = generate_window(window_dict, date0_min, date0_max)

    # perform experiment
    ret_e, ret_v = experiment(df_rich, textual, window_iter, model_name, perc_ls)

    return ret_e, ret_v
