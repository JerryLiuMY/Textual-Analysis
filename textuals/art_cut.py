from global_settings import RICH_PATH, DATA_PATH
from global_settings import stop_list
from global_settings import full_dict
from tools.text_tools import join_tt
from datetime import datetime
import pandas as pd
import pickle
import jieba
import math
import os
import re


def build_art_cut(sub_file_rich):
    """ compute textual for doc2vec
    :param sub_file_rich: enriched sub file
    """

    # load sub_df_rich
    textual_name = "art_cut"
    textual_path = os.path.join(DATA_PATH, textual_name)
    sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich))
    sub_df_rich["title"] = sub_df_rich["title"].astype(str)
    sub_df_rich["text"] = sub_df_rich["text"].astype(str)

    # configure jieba
    for word in full_dict:
        jieba.add_word(word)

    def cut_art(art): return [_ for _ in " ".join(jieba.cut(art, cut_all=False, HMM=True)).split()]

    # cut article
    mini_size = 100
    sub_art_cut = pd.Series(name="art_cut", dtype=object)

    for idx, iloc in enumerate(range(0, sub_df_rich.shape[0], mini_size)):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Working on {sub_file_rich} -- progress {idx + 1} / {math.ceil(sub_df_rich.shape[0] / mini_size)}")

        mini_df_rich = sub_df_rich.iloc[iloc: iloc + mini_size, :].reset_index(inplace=False, drop=True)
        mini_art_cut = mini_df_rich.apply(join_tt, axis=1).apply(cut_art)
        mini_art_cut = mini_art_cut.apply(lambda _: [w for w in _ if len(re.findall(r"[\u4e00-\u9fff]+", w)) != 0])
        mini_art_cut = mini_art_cut.apply(lambda _: [w for w in _ if w not in stop_list])
        mini_art_cut.name = "art_cut"
        sub_art_cut = sub_art_cut.append(mini_art_cut)

    sub_art_cut.reset_index(inplace=True, drop=True)
    sub_text_file = f"{sub_file_rich.split('.')[0]}.pkl"
    print(f"Saving to {sub_text_file}...")
    with open(os.path.join(textual_path, sub_text_file), "wb") as f:
        pickle.dump(sub_art_cut, f)
