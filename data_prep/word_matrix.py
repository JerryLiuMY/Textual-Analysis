import pandas as pd
import numpy as np
import os
from global_settings import DATA_PATH
from global_settings import RICH_PATH
from global_settings import WORD_PATH


# https://github.com/MengLingchao/Chinese_financial_sentiment_dictionary
xlsx_dict = pd.ExcelFile(os.path.join(DATA_PATH, "Chinese_Dict.xlsx"))
pos_dict = [_.strip() for _ in xlsx_dict.parse("positive").iloc[:, 0]]
neg_dict = [_.strip() for _ in xlsx_dict.parse("negative").iloc[:, 0]]
full_dict = pos_dict + neg_dict


def compute_matrix(sub_file_rich):
    """ compute word matrix
    :param sub_file_rich: enriched sub file
    """

    # load sub file enriched
    sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich))
    def func(df): df["text"] if df["title"] is np.nan else " ".join([df["title"], df["text"]])
    sub_df_rich["article"] = sub_df_rich.apply(func, axis=1)

    sub_word_matrix = sub_df_rich.loc[:, "article"].apply(lambda _: [_.count(word) for word in full_dict])
    sub_word_df = pd.DataFrame(sub_word_matrix.tolist(), columns=full_dict)

    sub_word_file = f"word_{sub_file_rich.split('.')[0].split('_')[1]}.csv"
    print(f"Saving to {sub_word_file}...")
    sub_word_df.to_csv(os.path.join(WORD_PATH, sub_word_file), index=False)
