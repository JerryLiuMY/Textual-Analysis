import os
import pickle
import datetime
import pandas as pd
import numpy as np
import scipy as sp
import functools
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
from experiments.generators import generate_window
from params.params import window_dict, date0_min, date0_max


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
    num_proc = 12
    for idx in range(0, len(sub_file_clean_li), num_proc):
        pool = Pool(num_proc)
        pool.map(enrich_data, sub_file_clean_li[idx: idx + num_proc])
        pool.close()
        pool.join()


def run_textual(textual_name):
    """ Build textual data
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
        raise ValueError("Mismatch between enriched data files and textual files")

    sub_file_rich_li = sorted([_.split("/")[-1] for _ in glob(os.path.join(RICH_PATH, "*.csv"))])
    sub_text_file_li = sorted([_.split("/")[-1] for _ in glob(os.path.join(text_path, extension))])

    return zip(sub_file_rich_li, sub_text_file_li)


def load_word_sps():
    """ Load word sparse matrix """

    # get df_rich & word_sps
    text_path = os.path.join(DATA_PATH, "word_sps")
    files_iter = generate_files("word_sps")
    df_rich = pd.DataFrame()
    word_sps = csr_matrix(np.empty((0, len(full_dict)), dtype=np.int64))

    for sub_file_rich, sub_text_file in files_iter:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Combining {sub_file_rich} and {sub_text_file}")
        sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich))
        sub_word_sps = load_npz(os.path.join(text_path, sub_text_file))
        df_rich = df_rich.append(sub_df_rich)
        word_sps = sp.sparse.vstack([word_sps, sub_word_sps], format="csr")

    df_rich.reset_index(inplace=True, drop=True)

    return df_rich, word_sps


def load_art_cut():
    """ Load articles cut with jieba """

    # get df_rich & art_cut
    text_path = os.path.join(DATA_PATH, "art_cut")
    files_iter = generate_files("art_cut")
    df_rich = pd.DataFrame()
    art_cut = pd.Series(dtype=object)

    for sub_file_rich, sub_text_file in files_iter:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Combining {sub_file_rich} and {sub_text_file}")
        sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich))
        with open(os.path.join(text_path, sub_text_file), "rb") as f:
            sub_art_cut = pickle.load(f)
        df_rich = df_rich.append(sub_df_rich)
        art_cut = pd.concat([art_cut, sub_art_cut], axis=0)

    art_cut.name = "art_cut"
    df_rich.reset_index(inplace=True, drop=True)
    art_cut.reset_index(inplace=True, drop=True)

    return df_rich, art_cut


def run_experiment(model_name, perc_ls):
    """ Run experiment
    :param model_name: model name
    :param perc_ls: percentage of L/S portfolio
    """

    # create model directory
    model_path = os.path.join(OUTPUT_PATH, model_name)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # create sub-directories
    for ev in ["e", "v"]:
        params_sub_path = os.path.join(model_path, f"params_{ev}")
        if not os.path.isdir(params_sub_path):
            os.mkdir(params_sub_path)

        model_sub_path = os.path.join(model_path, f"model_{ev}")
        if not os.path.isdir(model_sub_path):
            os.mkdir(model_sub_path)

    return_sub_path = os.path.join(model_path, "return")
    if not os.path.isdir(return_sub_path):
        os.mkdir(return_sub_path)

    # get df_rich & textual
    if model_name == "ssestm":
        df_rich, textual = load_word_sps()
    elif model_name in ["doc2vec", "bert"]:
        df_rich, textual = load_art_cut()
    else:
        raise ValueError("Invalid model name")

    # perform experiment
    num_proc = 12
    window_li = list(generate_window(window_dict, date0_min, date0_max))
    for idx in range(0, len(window_li), num_proc):
        pool = Pool(num_proc)
        func = functools.partial(experiment, df_rich=df_rich, textual=textual, model_name=model_name, perc_ls=perc_ls)
        pool.map(func, window_li[idx: idx + num_proc])
        pool.close()
        pool.join()

    # combine to get return
    ret_csv = pd.concat([pd.read_csv(_, index_col=0) for _ in sorted(glob(os.path.join(return_sub_path, "*.csv")))])
    ret_pkl = pd.concat([pd.read_pickle(_) for _ in sorted(glob(os.path.join(return_sub_path, "*.pkl")))])
    ret_csv.to_csv(os.path.join(model_path, "ret_csv.csv"))
    ret_pkl.to_pickle(os.path.join(model_path, "ret_pkl.pkl"))

    # backtest
    backtest(model_name, dalym)
