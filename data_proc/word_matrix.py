import pandas as pd
import numpy as np
import os
from global_settings import DATA_PATH
from global_settings import RICH_PATH
from global_settings import WORD_PATH
import datetime
import math

# https://github.com/MengLingchao/Chinese_financial_sentiment_dictionary
xlsx_dict = pd.ExcelFile(os.path.join(DATA_PATH, "Chinese_Dict.xlsx"))
pos_dict = [_.strip() for _ in xlsx_dict.parse("positive").iloc[:, 0]]
neg_dict = [_.strip() for _ in xlsx_dict.parse("negative").iloc[:, 0]]
full_dict = pos_dict + neg_dict


def build_word(sub_file_rich):
    """ compute word matrix
    :param sub_file_rich: enriched sub file
    """

    # load sub file enriched
    sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich))
    def func(df): return df["text"] if df["title"] is np.nan else " ".join([df["title"], df["text"]])

    # build word matrix
    mini_size = 100
    sub_word_df = pd.DataFrame()

    for idx, iloc in enumerate(range(0, sub_df_rich.shape[0], mini_size)):
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Working on {sub_file_rich} -- progress {idx + 1} / {math.ceil(sub_df_rich.shape[0] / mini_size)}")

        mini_df_rich = sub_df_rich.iloc[iloc: iloc + mini_size, :].reset_index(inplace=False, drop=True)
        mini_word_matrix = mini_df_rich.apply(func, axis=1).apply(lambda _: [_.count(word) for word in full_dict])
        mini_word_df = pd.DataFrame(mini_word_matrix.tolist(), columns=full_dict)
        sub_word_df = sub_word_df.append(mini_word_df)

    sub_word_file = f"word_{sub_file_rich.split('.')[0].split('_')[1]}.csv"
    print(f"Saving to {sub_word_file}...")
    sub_word_df.to_csv(os.path.join(WORD_PATH, sub_word_file), index=False)
