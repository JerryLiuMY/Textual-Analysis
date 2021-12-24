import os
import pandas as pd
import numpy as np
import scipy as sp
import pickle
import datetime
from glob import glob
from multiprocessing.pool import Pool
from scipy.sparse import load_npz, csr_matrix
from global_settings import CLEAN_PATH
from global_settings import DATA_PATH
from global_settings import RICH_PATH
from global_settings import OUTPUT_PATH
from global_settings import full_dict
from global_settings import dalym
from data_prep.data_clean import save_data
from data_prep.data_clean import clean_data
from data_prep.data_split import split_data
from data_prep.data_enrich import enrich_data
from textual.word_sps import build_word_sps
from textual.art_cut import build_art_cut
from experiments.experiment import experiment
from experiments.backtest import backtest


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

    # define index
    sub_file_clean_li = [_.split("/")[-1] for _ in glob(os.path.join(CLEAN_PATH, "*.csv"))]
    sub_file_rich_idx = [_.split("/")[-1].split(".")[0].split("_")[1] for _ in glob(os.path.join(RICH_PATH, "*.csv"))]
    sub_file_clean_li = sorted([_ for _ in sub_file_clean_li if _.split(".")[0].split("_")[1] not in sub_file_rich_idx])

    # enrich data
    num_proc = 16
    for idx in range(0, len(sub_file_clean_li), num_proc):
        pool = Pool(num_proc)
        pool.map(enrich_data, sub_file_clean_li[idx: idx + num_proc])
        pool.close()
        pool.join()


def run_textual(textual_name):
    """ Build word sparse matrix
    :param textual_name: textual name
    """

    # create path
    text_path = os.path.join(DATA_PATH, textual_name)
    if not os.path.isdir(text_path):
        os.mkdir(text_path)

    # define functions
    if textual_name == "word_sps":
        build_textual, extension = build_word_sps, "*.npz"
    elif textual_name == "art_cut":
        build_textual, extension = build_art_cut, "*.pkl"
    else:
        raise ValueError("Invalid textual name")

    # define paths
    sub_file_rich_li = [_.split("/")[-1] for _ in glob(os.path.join(RICH_PATH, "*.csv"))]
    sub_text_file_idx = [_.split("/")[-1].split(".")[0].split("_")[2] for _ in glob(os.path.join(text_path, extension))]
    sub_file_rich_li = sorted([_ for _ in sub_file_rich_li if _.split(".")[0].split("_")[1] not in sub_text_file_idx])

    # build textual
    num_proc = 12
    for idx in range(0, len(sub_file_rich_li), num_proc):
        pool = Pool(num_proc)
        pool.map(build_textual, sub_file_rich_li[idx: idx + num_proc])
        pool.close()
        pool.join()


def generate_files(textual_name):
    """ Build iterator for files """

    # define paths
    text_path = os.path.join(DATA_PATH, textual_name)
    extension = "*.npz" if textual_name == "word_sps" else "*.pkl"
    sub_file_rich_idx = [_.split("/")[-1].split(".")[0].split("_")[1] for _ in glob(os.path.join(RICH_PATH, "*.csv"))]
    sub_text_file_idx = [_.split("/")[-1].split(".")[0].split("_")[2] for _ in glob(os.path.join(text_path, extension))]
    if sorted(sub_file_rich_idx) != sorted(sub_text_file_idx):
        raise ValueError("Mismatch between enriched files and word matrix files")

    sub_file_rich_li = sorted([_.split("/")[-1] for _ in glob(os.path.join(RICH_PATH, "*.csv"))])
    sub_text_file_li = sorted([_.split("/")[-1] for _ in glob(os.path.join(text_path, extension))])

    return text_path, zip(sub_file_rich_li, sub_text_file_li)


def build_ssestm():
    """ Build experiment for ssestm """

    # get df_rich & word_sps
    df_rich = pd.DataFrame()
    word_sps = csr_matrix(np.empty((0, len(full_dict)), np.int64))
    text_path, files_iter = generate_files("word_sps")

    for sub_file_rich, sub_text_file in files_iter:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Combining {sub_file_rich} and {sub_text_file}")
        sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich))
        sub_word_sps = load_npz(os.path.join(text_path, sub_text_file))
        df_rich = df_rich.append(sub_df_rich)
        word_sps = sp.sparse.vstack([word_sps, sub_word_sps], format="csr")

    df_rich.reset_index(inplace=True, drop=True)

    return df_rich, word_sps


def build_doc2vec():
    """ Build experiment for doc2vec """

    # define index
    text_path = os.path.join(DATA_PATH, "art_cut")
    sub_file_rich_idx = [_.split("/")[-1].split(".")[0].split("_")[1] for _ in glob(os.path.join(RICH_PATH, "*.csv"))]
    sub_text_file_idx = [_.split("/")[-1].split(".")[0].split("_")[2] for _ in glob(os.path.join(text_path, "*.pkl"))]
    if sorted(sub_file_rich_idx) != sorted(sub_text_file_idx):
        raise ValueError("Mismatch between enriched files and word matrix files")

    sub_file_rich_li = sorted([_.split("/")[-1] for _ in glob(os.path.join(RICH_PATH, "*.csv"))])
    sub_text_file_li = sorted([_.split("/")[-1] for _ in glob(os.path.join(text_path, "*.pkl"))])

    # get df_rich & art_cut
    df_rich = pd.DataFrame()
    art_cut = pd.Series(dtype=object)
    for sub_file_rich, sub_text_file in zip(sub_file_rich_li, sub_text_file_li):
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Working on {sub_file_rich}")
        sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich))
        with open(os.path.join(text_path, sub_text_file), "wb") as f:
            sub_art_cut = pickle.load(f)
        df_rich = df_rich.append(sub_df_rich)
        art_cut = pd.concat([art_cut, sub_art_cut], axis=0)

    df_rich.reset_index(inplace=True, drop=True)
    art_cut.reset_index(inplace=True, drop=True)
    art_cut.name = "art_cut"

    return df_rich, art_cut


def build_bert():
    """ Build experiment for bert """

    return None, None


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
    elif model_name == "bert":
        df_rich, textual = build_bert()
    else:
        raise ValueError("Invalid model name")

    # perform experiment
    experiment(df_rich, textual, model_name, perc_ls)


def run_backtest(model_name):
    """ Run backtest
    :param model_name: model name
    """

    # backtest
    backtest(model_name, dalym)
