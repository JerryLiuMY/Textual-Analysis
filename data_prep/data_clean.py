import os
import pandas as pd
import json
from global_settings import DATA_PATH, LOG_PATH
from tools.utils import convert_datetime


def save_raw_data():
    """ i) load original data ii) select useful columns iii) convert timestamp to datetime and iv) save raw data
    :return: raw data dataframe
    """

    col_names = [
        "user_id",
        "comment_id",
        "created_at",
        "title",
        "text",
        "retweet_count",
        "reply_count",
        "fav_count",
        "like_count",
        "stock_corr",
        "stock_mention"
    ]

    print("Loading raw data...")
    data_df = pd.read_csv(os.path.join(DATA_PATH, "raw_data.csv"), names=col_names)

    print("Selecting useful columns...")
    data_df = data_df.loc[:, ["created_at", "text", "stock_mention"]]

    print("Converting to datetime...")
    datetime = data_df["created_at"].apply(convert_datetime)
    data_df["date"] = datetime.apply(lambda _: _[0])
    data_df["time"] = datetime.apply(lambda _: _[1])
    data_df = data_df.loc[:, data_df.columns != "created_at"]

    print("Saving data...")
    data_df.to_csv(os.path.join(DATA_PATH, "XueQiu.csv"), index=False)


def clean_data(data_df):
    """ Drop i) nan entries ii) entries with more than one stock mentioned iii) entries without csmar matches
    :param data_df: raw data dataframe
    :return: cleaned dataframe
    """

    with open(os.path.join(LOG_PATH, "data_log.json"), "r") as f:
        data_log = json.load(f)

    # original data
    data_log["original"] = data_df.shape[0]

    # drop NaN entries
    print("Dropping NaN entries...")
    data_df = data_df.dropna(subset=["stock_mention"], inplace=False)
    data_log["drop_nan"] = data_df.shape[0]

    # drop entries with more than one stock mentioned (and with other symbols)
    print("Dropping entries with more than one stock mentioned...")
    sym_li = ["|", "\\", "$", "(", ")"]
    data_df = data_df.loc[~data_df["stock_mention"].apply(lambda _: any(sym in _ for sym in sym_li)), :]
    data_log["single_tag"] = data_df.shape[0]

    # drop entries without matches with the csmar database
    print("Dropping entries without matches with the csmar database...")
    data_log["match_stkcd"] = data_df.shape[0]
    data_df.reset_index(inplace=False)

    # save log
    with open(os.path.join(LOG_PATH, "data_log.json"), "w") as f:
        json.dump(data_log, f)

    return data_df


def split_data(data_df):
    size = data_df.shape[0]
    sub_size = int(size / 100)

    for idx, start in enumerate(range(0, size, sub_size)):
        sub_data_df = data_df[start: start + sub_size]
        sub_path = os.path.join(DATA_PATH, "split", f"XueQiu_{idx}.csv")
        sub_data_df.to_csv(sub_path, index=False)


if __name__ == "__main__":
    from tools.log import init_data_log
    init_data_log()
    save_raw_data()
