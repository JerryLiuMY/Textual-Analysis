from tools.utils import shift_date, match_date
from tools.sql import query_dalyr
import pandas as pd
import os
import math
from global_settings import CLEAN_PATH


def enrich_data(sub_file):
    """ add i) trading date ii) type, cls, cap, ret iii) ret3 to the sub dataframe to be enriched
    :param sub_file: sub file to be enriched
    :return:
    """

    # load sub file
    sub_df = pd.read_csv(os.path.join(CLEAN_PATH, sub_file))

    # create auxiliary columns
    cls_time = "15:00:00"
    sub_df["shift"] = sub_df["time"].apply(lambda _: _[:2] >= cls_time).astype(int)
    sub_df["stkcd"] = sub_df["stock_mention"].apply(lambda _: _[2:])

    # match to the next trading date
    sub_df["date_0"] = sub_df.apply(lambda _: shift_date(_["date"], _["shift"]), axis=1).apply(match_date)

    # fetch type, cls, cap, ret
    sub_df_rich = pd.DataFrame()
    mini_size = 100
    for idx, iloc in enumerate(range(0, sub_df.shape[0], mini_size)):
        print(f"Working on {sub_file} -- current progress {idx + 1}/{math.ceil(sub_df.shape[0] / mini_size)}")
        mini_df = sub_df.iloc[iloc: iloc + mini_size, :]
        names = ["type", "cls", "cap", "ret"]
        result = mini_df.apply(lambda _: query_dalyr(_["stkcd"], _["date_0"]), axis=1)
        mini_df_rich = pd.concat([mini_df, pd.DataFrame(list(result), columns=names)])
        sub_df_rich = sub_df_rich.append(mini_df_rich, axis=1)

    return sub_df_rich
