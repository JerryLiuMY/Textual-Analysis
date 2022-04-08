from scipy.stats import rankdata
import matplotlib.pyplot as plt
from global_settings import LOG_PATH
from global_settings import RICH_PATH
import glob
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import os
sns.set()


def plot_year_count(data_df):
    """ Bar plot of the number of articles per year
    :param data_df: cleaned data_df
    """

    years = list(data_df["date"].apply(lambda _: int(_[:4])))
    max_year, min_year = max(years), min(years)
    x = np.arange(min_year, max_year + 1)
    height = [years.count(_) for _ in x]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.bar(x, height)
    ax.set_xticks(x)
    ax.set_xticklabels([str(_) for _ in x])
    ax.set_yticks(np.array([2, 4, 6, 8]) * 1e6)
    ax.set_yticklabels(["2M", "4M", "6M", "8M"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Num. Articles")

    fig.savefig(os.path.join(LOG_PATH, "yearly_count.pdf"), bbox_inches="tight")


def plot_day_count(data_df):
    """ Bar plot of the number of articles per day across the year
    :param data_df: cleaned data_df
    """

    days = list(data_df["date"].apply(lambda _: _[5:]))
    x = sorted(set(days))
    height = [days.count(_) for _ in tqdm(x)]
    xticks = [f"{_}".zfill(2) + "-01" for _ in np.arange(1, 13)]
    xticklabels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.bar(x, height, color="blue")
    plt.setp(ax.patches, linewidth=0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(np.array([2, 4, 6, 8, 10]) * 1e4)
    ax.set_yticklabels(["20k", "40k", "60k", "80k", "100k"])
    ax.set_xlabel("Month")
    ax.set_ylabel("Num. Articles")

    fig.savefig(os.path.join(LOG_PATH, "daily_count.pdf"), bbox_inches="tight")


def plot_hour_count(data_df):
    """ Bar plot of the number of articles per hour across the day
    :param data_df: cleaned data_df
    """

    time = list(data_df["time"].apply(lambda _: _[:5]))
    x = [str(f"{hour}".zfill(2) + ":" + minute) for hour in range(12) for minute in ["00", "30"]] + ["12:00"]
    bins = [(x[i], x[i + 1]) for i in range(len(x) - 1)]
    height = [sum([(_[0] <= t) and (t < _[1]) for t in time]) for _ in tqdm(bins)]

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.bar(np.arange(len(bins)) + 0.5, height, color=sns.color_palette()[1])
    ax.set_xticks(np.arange(len(x)))
    ax.set_xticklabels(x)
    ax.set_yticks(np.array([0.3, 0.6, 0.9, 1.2, 1.5]) * 1e6)
    ax.set_yticklabels(["0.3M", "0.6M", "0.9M", "1.2M", "1.5M"])
    ax.tick_params(axis="x", rotation=45)
    ax.set_xlabel("Month")
    ax.set_ylabel("Num. Articles")

    fig.savefig(os.path.join(LOG_PATH, "hourly_count.pdf"), bbox_inches="tight")


def plot_stock_count(sub_file_rich_li):
    """ Plot the number of stocks mentioned in articles
    :param sub_file_rich_li: list of enriched files
    """

    # define heights
    height = np.zeros(len(sub_file_rich_li))
    for i, file in enumerate(sub_file_rich_li):
        df_rich = pd.read_csv(file)
        height[i] = len(set(df_rich["stock_mention"]))

    # define xticks & xticklabels
    dates = [sub_file_rich.split("/")[-1].split(".")[0] for sub_file_rich in sub_file_rich_li]
    xticks = [date for date in dates if date[5:7] in ["01", "07"]]
    xticks = [xticks[0]] + [date for i, date in enumerate(xticks[1:]) if date[5:7] != xticks[i][5:7]]

    def xtick_to_xticklabel(xtick):
        year = xtick[:4]
        month = "Jan" if xtick[5:7] == "01" else "Jul"
        xticklabel = " ".join([month, year])

        return xticklabel

    xticklabels = [xtick_to_xticklabel(xtick) for xtick in xticks]

    # plot stock count
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.bar(dates, height, color="blue")
    plt.setp(ax.patches, linewidth=0)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    fig.savefig(os.path.join(LOG_PATH, "zd_ret.pdf"), bbox_inches="tight")
    ax.set_xlabel("Data")
    ax.set_ylabel("Num. Stocks Mentioned")

    fig.savefig(os.path.join(LOG_PATH, "stock_count.pdf"), bbox_inches="tight")


def plot_zd_ret(sub_file_rich_li):
    """ plot the average returns of stocks with "涨" and "跌" inside the enriched files
    :param sub_file_rich_li: list of enriched files
    """

    z_ret_li = np.zeros(len(sub_file_rich_li))
    d_ret_li = np.zeros(len(sub_file_rich_li))

    for i, file in enumerate(sub_file_rich_li):
        df_rich = pd.read_csv(file)
        # average return of stocks with "涨" in the text
        z_ret = df_rich["ret3"].values[df_rich["text"].apply(lambda _: "涨" in _)].mean()
        # average return of stocks with "跌" in the text
        d_ret = df_rich["ret3"].values[df_rich["text"].apply(lambda _: "跌" in _)].mean()
        z_ret_li[i], d_ret_li[i] = z_ret, d_ret

        # print(f"涨: {round(z_ret, 4)}, 跌: {round(d_ret, 4)}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(z_ret_li, label="涨", color="red", alpha=0.65, bins=50)
    ax.hist(d_ret_li, label="跌", color="green", alpha=0.65, bins=50)
    ax.legend()

    fig.savefig(os.path.join(LOG_PATH, "zd_ret.pdf"), bbox_inches="tight")


def plot_zd_rank(sub_file_rich_li):
    """ plot the ranks of average returns of stocks with "涨" and "跌" inside the enriched files
    :param sub_file_rich_li: list of enriched files
    """

    z_rank_li = np.zeros(len(sub_file_rich_li))
    d_rank_li = np.zeros(len(sub_file_rich_li))

    for i, file in enumerate(sub_file_rich_li):
        df_rich = pd.read_csv(file)
        n = df_rich["ret3"].shape[0]
        p_hat = (rankdata(df_rich["ret3"].values) - 1) / n
        # normalized rank of returns of stocks with "涨" in the text
        z_rank = p_hat[df_rich["text"].apply(lambda _: "涨" in _)].mean()
        # normalized rank of returns of stocks with "跌" in the text
        d_rank = p_hat[df_rich["text"].apply(lambda _: "跌" in _)].mean()
        z_rank_li[i], d_rank_li[i] = z_rank, d_rank

        # print(f"涨: {round(z_rank, 4)}, 跌: {round(d_rank, 4)}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(z_rank_li, label="涨", color="red", alpha=0.65, bins=50)
    ax.hist(d_rank_li, label="跌", color="green", alpha=0.65, bins=50)
    ax.legend()

    fig.savefig(os.path.join(LOG_PATH, "zd_rank.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    sub_file_rich_li = sorted(glob.glob(os.path.join(RICH_PATH, "*.csv")))
    plot_stock_count(sub_file_rich_li)
    plot_zd_ret(sub_file_rich_li)
    plot_zd_rank(sub_file_rich_li)
