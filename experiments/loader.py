from scipy.sparse import load_npz
from experiments.generators import generate_files
from global_settings import DATA_PATH, RICH_PATH
from datetime import datetime
import pandas as pd
import psutil
import os


def input_loader(trddt, textual_name):
    """ Load input for experiment
    :param trddt: list of trddt dates
    :param textual_name: textual name
    """

    # get df_rich & word_sps
    sub_file_li = list(generate_files(trddt, textual_name))
    sub_file_rich_li = [_[0] for _ in sub_file_li]
    sub_text_file_li = [_[1] for _ in sub_file_li]

    # build df_rich
    columns = ["date_0", "ret3", "stock_mention", "ret", "cap"]
    df_rich = pd.DataFrame(columns=columns)
    for sub_file_rich in sub_file_rich_li:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Combining {sub_file_rich} "
              f"({psutil.virtual_memory().percent}% mem used)")

        sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich), low_memory=False)
        df_rich = df_rich.append(sub_df_rich.loc[:, columns])

    df_rich = df_rich.reset_index(inplace=False, drop=True)
    textual = generate_textual(textual_name, sub_text_file_li)

    return df_rich, textual


def generate_textual(textual_name, sub_text_file_li):
    """ Generate textual data
    :param textual_name: textual name
    :param sub_text_file_li: list of textual files
    """

    textual_path = os.path.join(DATA_PATH, textual_name)
    textual_loader = load_npz if textual_name == "word_sps" else pd.read_pickle

    doing = "Loading" if len(sub_text_file_li) == 1 else "Generating"
    for sub_text_file in sub_text_file_li:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"{doing} {sub_text_file} "
              f"({psutil.virtual_memory().percent}% mem used)")

        sub_textual = textual_loader(os.path.join(textual_path, sub_text_file))

        yield sub_textual
