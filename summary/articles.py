import os
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from global_settings import LOG_PATH
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
    """Bar plot of the number of articles per hour across the day
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
