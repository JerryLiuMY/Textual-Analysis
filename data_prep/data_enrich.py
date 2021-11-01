from tools.utils import shift_date
from tools.utils import match_date


def enrich_data(sub_df):
    """ add i) trading date ii) mkt_type, cls, mkt_cap, ret to the sub dataframe to be enriched
    :param sub_df: sub_df to be enriched
    :return:
    """

    # match to the next trading date
    sub_df["shift"] = sub_df["time"].apply(lambda _: _[:2] >= "15:00").astype(int)
    sub_df["date_0"] = sub_df.apply(lambda _: shift_date(_["date"], _["shift"]), axis=1).apply(match_date)

    # fetch mkt_type, cls, cap, ret for the particular date

    return sub_df

