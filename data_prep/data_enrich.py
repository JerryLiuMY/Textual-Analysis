from tools.data_tools import shift_date, match_date
from tools.sql import query_dalyr
from global_settings import CLEAN_PATH, RICH_PATH
from global_settings import user, host, password
import mysql.connector
from datetime import datetime
import pandas as pd
import math
import os


def enrich_data(sub_file_clean):
    """ add i) trading date ii) type, cls, cap, ret iii) ret3 to the sub dataframe to be enriched
    :param sub_file_clean: sub file to be enriched
    """

    # load sub file
    sub_df_clean = pd.read_csv(os.path.join(CLEAN_PATH, sub_file_clean))

    # create auxiliary columns
    cls_time = "15:00:00"
    sub_df_clean["shift"] = sub_df_clean["time"].apply(lambda _: _[:2] >= cls_time).astype(int)
    sub_df_clean["stkcd"] = sub_df_clean["stock_mention"].apply(lambda _: _[2:])
    sub_df_clean["date_t"] = sub_df_clean.apply(lambda _: shift_date(_["date"], _["shift"]), axis=1)

    # match to the trading date t, t-2, t+1
    sub_df_clean["date_0"] = sub_df_clean["date_t"].apply(lambda _: match_date(_, match_day=0))
    sub_df_clean["date_p1"] = sub_df_clean["date_t"].apply(lambda _: match_date(_, match_day=1))
    sub_df_clean["date_m2"] = sub_df_clean["date_t"].apply(lambda _: match_date(_, match_day=-2))

    # fetch type, cls, cap, ret, ret3
    mini_size = 100
    sub_df_rich = pd.DataFrame()

    csmar = mysql.connector.connect(user=user, password=password, host=host, database="CSMAR")
    for idx, iloc in enumerate(range(0, sub_df_clean.shape[0], mini_size)):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Working on {sub_file_clean} -- progress {idx + 1} / {math.ceil(sub_df_clean.shape[0] / mini_size)}")
        cursor = csmar.cursor()
        mini_df_clean = sub_df_clean.iloc[iloc: iloc + mini_size, :].reset_index(inplace=False, drop=True)
        result = mini_df_clean.apply(lambda _: query_dalyr(cursor, _["stkcd"], _["date_0"], select="all"), axis=1)
        cls_p1 = mini_df_clean.apply(lambda _: query_dalyr(cursor, _["stkcd"], _["date_p1"], select="CLSPRC"), axis=1)
        cls_m2 = mini_df_clean.apply(lambda _: query_dalyr(cursor, _["stkcd"], _["date_m2"], select="CLSPRC"), axis=1)
        result_df = pd.DataFrame(list(result), columns=["type", "cls", "cap", "ret"])

        cls_p1_df = pd.DataFrame(list(cls_p1))
        cls_m2_df = pd.DataFrame(list(cls_m2))
        ret3_df = cls_p1_df / cls_m2_df - 1.0
        ret3_df.columns = ["ret3"]

        mini_df_rich = pd.concat([mini_df_clean, result_df, ret3_df], axis=1)
        sub_df_rich = sub_df_rich.append(mini_df_rich)
        cursor.close()

    csmar.close()

    sub_df_rich.drop(columns=["shift", "stkcd", "date_t", "date_p1", "date_m2"], inplace=True)
    sub_df_rich.dropna(subset=["type", "cls", "cap", "ret", "ret3"], axis=0, inplace=True)
    sub_df_rich.reset_index(inplace=True, drop=True)

    sub_file_rich = f"enriched_{sub_file_clean.split('.')[0].split('_')[1]}.csv"
    print(f"Saving to {sub_file_rich}...")
    sub_df_rich.to_csv(os.path.join(RICH_PATH, sub_file_rich), index=False)
