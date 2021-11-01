from tools.utils import shift_date


def get_date_0(ticker, date):
    """
    :return:
    """

    sub_df["shift"] = sub_df["time"].apply(lambda _: _[:2] >= "15:00").astype(int)
    sub_df.apply(lambda _: shift_date(_["date"], _["shift"]), axis=1)

    return None
