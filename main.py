from scipy.sparse import load_npz, csr_matrix
from tools.exp_tools import get_df_rich, get_textual
from global_settings import DATA_PATH
from global_settings import RICH_PATH
from global_settings import full_dict
from datetime import datetime
from glob import glob
import pandas as pd
import numpy as np
import scipy as sp
import psutil
import pickle
import os


def generate_files(textual_name):
    """ Build iterator for files
    :param textual_name: name of textual model
    """

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
    columns = ["date_0", "ret3", "stock_mention", "ret", "cap"]
    df_rich = pd.DataFrame(columns=columns)
    word_sps = csr_matrix(np.empty((0, len(full_dict)), dtype=np.int64))

    for sub_file_rich, sub_text_file in files_iter:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Combining {sub_file_rich} and {sub_text_file} "
              f"({psutil.virtual_memory().percent}% mem used)")

        sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich))
        sub_word_sps = load_npz(os.path.join(text_path, sub_text_file))
        df_rich = df_rich.append(sub_df_rich.loc[:, columns])
        word_sps = sp.sparse.vstack([word_sps, sub_word_sps], format="csr")

    df_rich.reset_index(inplace=True, drop=True)

    return df_rich, word_sps


def load_art_cut():
    """ Load articles cut with jieba """

    # get df_rich & art_cut
    text_path = os.path.join(DATA_PATH, "art_cut")
    files_iter = generate_files("art_cut")
    columns = ["date_0", "ret3", "stock_mention", "ret", "cap"]
    df_rich = pd.DataFrame(columns=columns)
    art_cut = pd.Series(dtype=object)

    for sub_file_rich, sub_text_file in files_iter:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Combining {sub_file_rich} and {sub_text_file} "
              f"({psutil.virtual_memory().percent}% mem used)")

        sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich))
        with open(os.path.join(text_path, sub_text_file), "rb") as f:
            sub_art_cut = pickle.load(f)
        df_rich = df_rich.append(sub_df_rich.loc[:, columns])
        art_cut = pd.concat([art_cut, sub_art_cut], axis=0)

    art_cut.name = "art_cut"
    df_rich.reset_index(inplace=True, drop=True)
    art_cut.reset_index(inplace=True, drop=True)

    return df_rich, art_cut


def build_inputs(window, df_rich, textual):
    """ Load inputs for an experiment window
    :param window: window list
    :param df_rich: enriched dataframe
    :param textual: textual data
    """

    # build inputs
    trddt_train, trddt_valid, trddt_test = window
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
          f"Building inputs for {trddt_train[0][:-3]} to {trddt_test[-1][:-3]} "
          f"({psutil.virtual_memory().percent}% mem used)")

    trddt_window = trddt_train + trddt_valid + trddt_test
    window_idx = df_rich["date_0"].apply(lambda _: _ in trddt_window)
    df_rich_win = get_df_rich(df_rich, window_idx)
    textual_win = get_textual(textual, window_idx)

    return df_rich_win, textual_win
