from scipy.sparse import csr_matrix, save_npz
from global_settings import full_dict, RICH_PATH, DATA_PATH
from tools.text_tools import join_tt
from datetime import datetime
import pandas as pd
import numpy as np
import math
import os


def build_word_sps(sub_file_rich):
    """ compute word count sparse matrix
    :param sub_file_rich: enriched sub file
    """

    # load sub_df_rich
    textual_name = "word_sps"
    text_path = os.path.join(DATA_PATH, textual_name)
    sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich))
    sub_df_rich["title"] = sub_df_rich["title"].astype(str)
    sub_df_rich["text"] = sub_df_rich["text"].astype(str)

    # build word matrix
    mini_size = 100
    sub_word_mtx = np.empty([0, len(full_dict)])

    for idx, iloc in enumerate(range(0, sub_df_rich.shape[0], mini_size)):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Working on {sub_file_rich} -- progress {idx + 1} / {math.ceil(sub_df_rich.shape[0] / mini_size)}")

        mini_df_rich = sub_df_rich.iloc[iloc: iloc + mini_size, :].reset_index(inplace=False, drop=True)
        mini_word_mtx = mini_df_rich.apply(join_tt, axis=1).apply(lambda _: [_.count(word) for word in full_dict])
        sub_word_mtx = np.vstack([sub_word_mtx, np.array(mini_word_mtx.tolist())])

    sub_word_sps = csr_matrix(sub_word_mtx)
    sub_text_file = f"{textual_name}_{sub_file_rich.split('.')[0].split('_')[1]}.npz"
    print(f"Saving to {sub_text_file}...")
    save_npz(os.path.join(text_path, sub_text_file), sub_word_sps)
