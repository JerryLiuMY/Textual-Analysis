import os
import numpy as np
import pandas as pd
from glob import glob
import seaborn as sns
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from global_settings import RICH_PATH
from global_settings import LOG_PATH
sns.set()


def plot_zd_ret(rich_files):
    """ plot the average returns of stocks with "涨" and "跌" inside the enriched files
    :param rich_files: list of enriched files
    """

    z_ret_li = np.zeros(len(rich_files))
    d_ret_li = np.zeros(len(rich_files))

    for i, file in enumerate(rich_files):
        df_rich = pd.read_csv(file)
        # average return of stocks with "涨" in the text
        z_ret = df_rich["ret3"].values[df_rich["text"].apply(lambda _: "涨" in _)].mean()
        # average return of stocks with "跌" in the text
        d_ret = df_rich["ret3"].values[df_rich["text"].apply(lambda _: "跌" in _)].mean()
        z_ret_li[i], d_ret_li[i] = z_ret, d_ret

        # print(f"涨: {round(z_ret, 4)}, 跌: {round(d_ret, 4)}")

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.hist(z_ret_li, label="zhang", color="red", alpha=0.65, bins=50)
    ax.hist(d_ret_li, label="die", color="green", alpha=0.65, bins=50)
    ax.legend()

    fig.savefig(os.path.join(LOG_PATH, "zd_ret.pdf"), bbox_inches="tight")


def plot_zd_rank(rich_files):
    """ plot the ranks of average returns of stocks with "涨" and "跌" inside the enriched files
    :param rich_files: list of enriched files
    """

    z_rank_li = np.zeros(len(rich_files))
    d_rank_li = np.zeros(len(rich_files))

    for i, file in enumerate(rich_files):
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
    ax.hist(z_rank_li, label="zhang", color="red", alpha=0.65, bins=50)
    ax.hist(d_rank_li, label="die", color="green", alpha=0.65, bins=50)
    ax.legend()

    fig.savefig(os.path.join(LOG_PATH, "zd_rank.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    sub_file_rich_li = glob(os.path.join(RICH_PATH, "*.csv"))
    plot_zd_ret(sub_file_rich_li)
    plot_zd_rank(sub_file_rich_li)
