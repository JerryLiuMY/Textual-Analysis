from tools.utils import shift_date, match_date


def enrich_data(sub_df):
    """ add i) trading date ii) mkt_type, cls, mkt_cap, ret to the sub dataframe to be enriched
    :param sub_df: sub_df to be enriched
    :return:
    """

    # match to the next trading date
    cls_time = "15:00:00"
    sub_df["shift"] = sub_df["time"].apply(lambda _: _[:2] >= cls_time).astype(int)
    sub_df["date_0"] = sub_df.apply(lambda _: shift_date(_["date"], _["shift"]), axis=1).apply(match_date)

    # fetch type, cls, cap, ret

    return sub_df

