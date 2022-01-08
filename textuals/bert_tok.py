from global_settings import RICH_PATH, DATA_PATH
import official.nlp.bert.tokenization
from global_settings import stop_list
from datetime import datetime
from official.nlp import bert
import jieba
import pandas as pd
import pickle
import math
import os
import re


def build_bert_tok(sub_file_rich):
    """ compute word count sparse matrix
    :param sub_file_rich: enriched sub file
    """

    # load sub_df_rich
    textual_name = "bert_tok"
    textual_path = os.path.join(DATA_PATH, textual_name)
    sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich))
    sub_df_rich["title"] = sub_df_rich["title"].astype(str)
    sub_df_rich["text"] = sub_df_rich["text"].astype(str)
    tokenizer = bert.tokenization.FullTokenizer(vocab_file=os.path.join(textual_path, "pre-trained", "vocab.txt"))
    def join_tt(df): return df["text"] if df["title"] == "nan" else " ".join([df["title"], df["text"]])

    def tokenize(art):
        ids = []
        for line in re.findall(u"[^。！？!?]+[。！？!?]?", art, flags=re.U):
            sep_str = ["，", "。", "！", "？", "!", "?"]
            line = [_ for _ in " ".join(jieba.cut(line, cut_all=False, HMM=True)).split()]
            line = [_ for _ in line if _ not in [s for s in stop_list if s not in sep_str]]
            line = [_ for _ in line if (len(re.findall(r"[\u4e00-\u9fff]+", _)) != 0 or (_ in sep_str))]
            line = tokenizer.tokenize("".join([_ for _ in line])) + ["[SEP]"]
            ids.append(tokenizer.convert_tokens_to_ids(line))
        ids = [_ for sub_ids in ids for _ in sub_ids]

        return ids

    # build word matrix
    mini_size = 100
    sub_bert_tok = pd.Series(name="bert_tok", dtype=object)

    for idx, iloc in enumerate(range(0, sub_df_rich.shape[0], mini_size)):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Working on {sub_file_rich} -- progress {idx + 1} / {math.ceil(sub_df_rich.shape[0] / mini_size)}")

        mini_df_rich = sub_df_rich.iloc[iloc: iloc + mini_size, :].reset_index(inplace=False, drop=True)
        mini_bert_tok = mini_df_rich.apply(join_tt, axis=1).apply(tokenize)
        mini_bert_tok.name = "bert_tok"
        sub_bert_tok = sub_bert_tok.append(mini_bert_tok)

    sub_bert_tok.reset_index(inplace=True, drop=True)
    sub_text_file = f"{sub_file_rich.split('.')[0]}.pkl"
    print(f"Saving to {sub_text_file}...")
    with open(os.path.join(textual_path, sub_text_file), "wb") as f:
        pickle.dump(sub_bert_tok, f)
