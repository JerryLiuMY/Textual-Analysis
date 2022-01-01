import os
import pandas as pd
from global_settings import DATA_PATH, CLEAN_PATH
from global_settings import trddt_all, date0_min, date0_max
from tools.data_tools import shift_date, match_date
from datetime import datetime


def split_data(clean_file):
    """ Split the cleaned data
    :param clean_file: name of the cleaned file
    """

    # load cleaned file
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
          f"Loading {clean_file}...")
    clean_df = pd.read_csv(os.path.join(DATA_PATH, clean_file))

    # create auxiliary columns
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
          f"Creating date_t information...")
    cls_time = "15:00:00"
    clean_df["shift"] = clean_df["time"].apply(lambda _: _[:2] >= cls_time).astype(int)
    clean_df["date_t"] = clean_df.apply(lambda _: shift_date(_["date"], _["shift"]), axis=1)

    # match to the trading date t, t-2, t+1
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
          f"Creating date_0 information...")
    clean_df["date_0"] = clean_df["date_t"].apply(lambda _: match_date(_, match_day=0))
    clean_df["date_p1"] = clean_df["date_t"].apply(lambda _: match_date(_, match_day=1))
    clean_df["date_m2"] = clean_df["date_t"].apply(lambda _: match_date(_, match_day=-2))

    for trddt in trddt_all[(date0_min <= trddt_all) & (trddt_all <= date0_max)]:
        sub_df_clean = clean_df.loc[clean_df["date_0"].apply(lambda _: _ == trddt), :]
        sub_df_clean.reset_index(inplace=True, drop=True)

        sub_file_clean = f"{trddt}.csv"
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Saving to {sub_file_clean}...")
        sub_df_clean.to_csv(os.path.join(CLEAN_PATH, sub_file_clean), index=False)
