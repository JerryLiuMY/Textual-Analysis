from scipy.sparse import csr_matrix, load_npz
from experiments.generators import generate_files
from global_settings import DATA_PATH, RICH_PATH, full_dict
from datetime import datetime
import pickle
import numpy as np
import pandas as pd
import psutil
import scipy as sp
import os


def load_word_sps(trddt, textual_name):
    """ Load word sparse matrix
    :param trddt: list of trddt dates
    :param textual_name: textual name
    """

    # get df_rich & word_sps
    text_path = os.path.join(DATA_PATH, textual_name)
    files_li = list(generate_files(trddt, textual_name))

    if textual_name == "word_sps":
        load_func = load_npz
        textual = csr_matrix(np.empty((0, len(full_dict)), dtype=np.int64))
        def append_func(_): return sp.sparse.vstack(_, format="csr")
        def reset_func(_): return _
    elif textual_name == "art_cut":
        load_func = pd.read_pickle
        textual = pd.Series(name="art_cut", dtype=object)
        def append_func(_): return pd.concat(_, axis=0)
        def reset_func(_): return _.reset_index(inplace=False, drop=True)
    else:
        raise ValueError("Invalid textual name")

    # single file case
    doing = "Loading" if len(trddt) == 1 else "Combining"
    columns = ["date_0", "ret3", "stock_mention", "ret", "cap"]
    df_rich = pd.DataFrame(columns=columns)

    for sub_file_rich, sub_text_file in files_li:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"{doing} {sub_file_rich} and {sub_text_file} "
              f"({psutil.virtual_memory().percent}% mem used)")

        sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich), low_memory=False)
        sub_textual = load_func(os.path.join(text_path, sub_text_file))
        df_rich = df_rich.append(sub_df_rich.loc[:, columns])
        textual = append_func([textual, sub_textual])

    df_rich = df_rich.reset_index(inplace=False, drop=True)
    textual = reset_func(textual)

    return df_rich, textual
