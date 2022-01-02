from scipy.sparse import csr_matrix, load_npz
from experiments.generators import generate_files
from global_settings import DATA_PATH, RICH_PATH, full_dict
from datetime import datetime
import os
import pickle
import numpy as np
import pandas as pd
import psutil
import scipy as sp


def load_word_sps(trddt):
    """ Load word sparse matrix
    :param trddt: list of trddt dates
    """

    # get df_rich & word_sps
    text_path = os.path.join(DATA_PATH, "word_sps")
    files_iter = generate_files(trddt, "word_sps")
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


def load_art_cut(trddt):
    """ Load articles cut with jieba
    :param trddt: list of trddt dates
    """

    # get df_rich & art_cut
    text_path = os.path.join(DATA_PATH, "art_cut")
    files_iter = generate_files(trddt, "art_cut")
    columns = ["date_0", "ret3", "stock_mention", "ret", "cap"]
    df_rich = pd.DataFrame(columns=columns)
    art_cut = pd.Series(name="art_cut", dtype=object)

    for sub_file_rich, sub_text_file in files_iter:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Combining {sub_file_rich} and {sub_text_file} "
              f"({psutil.virtual_memory().percent}% mem used)")

        sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich))
        with open(os.path.join(text_path, sub_text_file), "rb") as f:
            sub_art_cut = pickle.load(f)
        df_rich = df_rich.append(sub_df_rich.loc[:, columns])
        art_cut = art_cut.append(sub_art_cut)

    df_rich.reset_index(inplace=True, drop=True)
    art_cut.reset_index(inplace=True, drop=True)

    return df_rich, art_cut


def load_input(trddt, textual_name):
    """ Load inputs
    :param trddt: list of trddt dates
    :param textual_name: textual name
    """

    if textual_name == "word_sps":
        df_rich, textual = load_word_sps(trddt)
    elif textual_name == "art_cut":
        df_rich, textual = load_art_cut(trddt)
    else:
        raise ValueError("Invalid textual name")

    return df_rich, textual
