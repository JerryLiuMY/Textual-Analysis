from experiments.generators import generate_files
from global_settings import DATA_PATH, RICH_PATH
from tools.exp_tools import iterable_wrapper
from params.params import subset_size
from scipy.sparse import load_npz
from scipy.sparse import issparse
from datetime import datetime
import pandas as pd
import numpy as np
import psutil
import os


def input_loader(trddt, textual_name, subset):
    """ Load input for experiment
    :param trddt: list of trddt dates
    :param textual_name: textual name
    :param subset: whether to use a subset of data
    """

    # get df_rich & textual
    sub_file_li = list(generate_files(trddt, textual_name))
    sub_file_rich_li = [_[0] for _ in sub_file_li]
    sub_text_file_li = [_[1] for _ in sub_file_li]
    sub_sampler_li = list()

    # build df_rich
    np.random.seed(10)
    columns = ["date_0", "ret3", "stock_mention", "ret", "cap"]
    df_rich = pd.DataFrame(columns=columns)
    for sub_file_rich in sub_file_rich_li:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Combining {sub_file_rich} "
              f"[{psutil.virtual_memory().percent}% mem] "
              f"({psutil.cpu_percent()}% cpu)")

        sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich), low_memory=False)
        sub_df_rich = sub_df_rich.loc[:, columns]

        if subset:
            sub_df_size = sub_df_rich.shape[0]
            sub_sampler = np.random.choice(sub_df_size, int(sub_df_size * subset_size), replace=False)
            sub_df_rich = sub_df_rich.iloc[sub_sampler, :].reset_index(inplace=False, drop=True)
            sub_sampler_li.append(sub_sampler)

        df_rich = df_rich.append(sub_df_rich)

    df_rich.reset_index(inplace=True, drop=True)
    textual = generate_textual(textual_name, sub_text_file_li, sub_sampler_li)

    return df_rich, textual


@iterable_wrapper
def generate_textual(textual_name, sub_text_file_li, sub_sampler_li):
    """ Generate textual data
    :param textual_name: textual name
    :param sub_text_file_li: list of textual files
    :param sub_sampler_li: list of samplers
    """

    # build textual
    textual_path = os.path.join(DATA_PATH, textual_name)
    if textual_name == "word_sps":
        textual_loader = load_npz
    elif textual_name == "art_cut":
        textual_loader = pd.read_pickle
    elif textual_name == "bert_tok":
        textual_loader = pd.read_pickle
    else:
        raise ValueError("Invalid textual name")

    doing = "Loading" if len(sub_text_file_li) == 1 else "Iterating"
    for idx, sub_text_file in enumerate(sub_text_file_li):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"{doing} {sub_text_file} "
              f"[{psutil.virtual_memory().percent}% mem] "
              f"({psutil.cpu_percent()}% cpu)")

        sub_textual = textual_loader(os.path.join(textual_path, sub_text_file))

        if len(sub_sampler_li) != 0:
            sub_sampler = sub_sampler_li[idx]
            sub_textual = slice_textual(sub_textual, sub_sampler)

        yield sub_textual


def slice_textual(textual, sampler):
    """ Get textual data from array of indices
    :param textual: textual data
    :param sampler: array of indices
    """

    if isinstance(textual, np.ndarray) or issparse(textual):
        return textual[sampler]
    elif isinstance(textual, pd.DataFrame) or isinstance(textual, pd.Series):
        return textual.iloc[sampler].reset_index(inplace=False, drop=True)
    else:
        raise ValueError("Textual type not recognized")
