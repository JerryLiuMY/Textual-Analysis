import os
import jieba
import pandas as pd
import numpy as np
import scipy as sp
import datetime
from glob import glob
from multiprocessing.pool import Pool
from scipy.sparse import load_npz, csr_matrix
from global_settings import CLEAN_PATH, date0_min, date0_max
from global_settings import RICH_PATH
from global_settings import WORD_PATH
from global_settings import OUTPUT_PATH
from global_settings import full_dict
from global_settings import dalym
from global_settings import stop_list
from data_prep.data_clean import save_data
from data_prep.data_clean import clean_data
from data_prep.data_split import split_data
from data_prep.data_enrich import enrich_data
from embeddings.word_sps import build_word_sps
from experiments.params import window_dict
from experiments.generators import generate_window
from experiments.experiment import experiment
from analysis.backtest import backtest


def create_dirs(paths):
    """ Create directories
    :param paths: directories to create
    """

    # create directories
    for path in paths:
        if not os.path.isdir(path):
            os.mkdir(path)


def run_data_prep(raw_file="raw.csv", data_file="data.csv", clean_file="cleaned.csv"):
    """ Run data processing -- clean, split & enrich data
    :param raw_file: raw file
    :param data_file: data file
    :param clean_file: cleaned file
    """

    # clean & split data
    split_num = 750
    save_data(raw_file, data_file)
    clean_data(data_file, clean_file)
    split_data(clean_file, split_num=split_num)

    # enrich data
    sub_file_clean_li = [_.split("/")[-1] for _ in glob(os.path.join(CLEAN_PATH, "*.csv"))]
    sub_file_rich_idx = [_.split("/")[-1].split(".")[0].split("_")[1] for _ in glob(os.path.join(RICH_PATH, "*.csv"))]
    sub_file_clean_li = sorted([_ for _ in sub_file_clean_li if _.split(".")[0].split("_")[1] not in sub_file_rich_idx])

    num_proc = 16
    for idx in range(0, len(sub_file_clean_li), num_proc):
        pool = Pool(num_proc)
        pool.map(enrich_data, sub_file_clean_li[idx: idx + num_proc])
        pool.close()
        pool.join()


def run_word_sps():
    """Build word sparse matrix"""

    # define index
    sub_file_rich_li = [_.split("/")[-1] for _ in glob(os.path.join(RICH_PATH, "*.csv"))]
    sub_word_file_idx = [_.split("/")[-1].split(".")[0].split("_")[1] for _ in glob(os.path.join(WORD_PATH, "*.npz"))]
    sub_file_rich_li = sorted([_ for _ in sub_file_rich_li if _.split(".")[0].split("_")[1] not in sub_word_file_idx])

    # build word sparse matrix
    num_proc = 12
    for idx in range(0, len(sub_file_rich_li), num_proc):
        pool = Pool(num_proc)
        pool.map(build_word_sps, sub_file_rich_li[idx: idx + num_proc])
        pool.close()
        pool.join()


def build_ssestm():
    """ Build experiment for ssestm"""

    # define index
    sub_file_rich_idx = [_.split("/")[-1].split(".")[0].split("_")[1] for _ in glob(os.path.join(RICH_PATH, "*.csv"))]
    sub_word_file_idx = [_.split("/")[-1].split(".")[0].split("_")[1] for _ in glob(os.path.join(WORD_PATH, "*.npz"))]
    if sorted(sub_file_rich_idx) != sorted(sub_word_file_idx):
        raise ValueError("Mismatch between enriched files and word matrix files")

    sub_file_rich_li = sorted([_.split("/")[-1] for _ in glob(os.path.join(RICH_PATH, "*.csv"))])
    sub_word_file_li = sorted([_.split("/")[-1] for _ in glob(os.path.join(WORD_PATH, "*.npz"))])

    # get df_rich & word_sps
    df_rich = pd.DataFrame()
    word_sps = csr_matrix(np.empty((0, len(full_dict)), np.int64))
    for sub_file_rich, sub_word_file in zip(sub_file_rich_li, sub_word_file_li):
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Combining {sub_file_rich} and {sub_word_file}")
        sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich))
        sub_word_sps = load_npz(os.path.join(WORD_PATH, sub_word_file))
        df_rich = df_rich.append(sub_df_rich)
        word_sps = sp.sparse.vstack([word_sps, sub_word_sps], format="csr")

    df_rich.reset_index(inplace=True, drop=True)

    return df_rich, word_sps


def build_doc2vec():
    """ Build experiment for doc2vec"""

    # define index
    sub_file_rich_li = sorted([_.split("/")[-1] for _ in glob(os.path.join(RICH_PATH, "*.csv"))])
    def join_tt(df): return df["text"] if df["title"] == "nan" else " ".join([df["title"], df["text"]])
    def cut_doc(doc): return [word for word in " ".join(jieba.cut(doc)).split() if word not in stop_list]

    # get df_rich & doc_cut
    df_rich = pd.DataFrame()
    doc_cut = pd.Series()
    for sub_file_rich in sub_file_rich_li:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Working on {sub_file_rich}")
        sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich))
        sub_df_rich["title"] = sub_df_rich["title"].astype(str)
        sub_df_rich["text"] = sub_df_rich["text"].astype(str)
        sub_doc_cut = sub_df_rich.apply(join_tt, axis=1).apply(cut_doc)
        df_rich = df_rich.append(sub_df_rich)
        doc_cut = pd.concat([doc_cut, sub_doc_cut], axis=0)

    df_rich.reset_index(inplace=True, drop=True)
    doc_cut.reset_index(inplace=True, drop=True)
    doc_cut.name = "doc_cut"

    return df_rich, doc_cut


def run_experiment(model_name, perc_ls):
    """ Run experiment
    :param model_name: model name
    :param perc_ls: percentage of L/S portfolio
    """

    # create path
    model_path = os.path.join(OUTPUT_PATH, model_name)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # get df_rich & textual
    if model_name == "ssestm":
        df_rich, textual = build_ssestm()
    elif model_name == "doc2vec":
        df_rich, textual = build_doc2vec()
    else:
        raise ValueError("Invalid model name")

    # rolling window
    window_iter = generate_window(window_dict, date0_min, date0_max)

    # perform experiment
    experiment(df_rich, textual, window_iter, model_name, perc_ls)


def run_backtest(model_name):
    """ Run backtest
    :param model_name: model name
    """

    # backtest
    backtest(model_name, dalym)
