import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
sns.set()


def plot_day_count(data_df):
    """ Bar plot of the number of articles per day across the year
    :param data_df: cleaned data_df
    :return: bar plot
    """
    days = sorted(list(data_df["date"].apply(lambda _: _[5:])))
    x = sorted(set(days))
    height = [days.count(_) for _ in tqdm(x)]
    xticks = [f"{_}".zfill(2) + "-01" for _ in np.arange(1, 13)]
    xticklabels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.bar(x, height)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_xlabel("Day")
    ax.set_ylabel("Num. Articles")
    fig.show()

    return fig


def plot_year_count(data_df):
    """ Bar plot of the number of articles per year
    :param data_df: cleaned data_df
    :return: bar plot
    """
    years = list(data_df["date"].apply(lambda _: int(_[:4])))
    max_year, min_year = max(years), min(years)
    x = np.arange(min_year, max_year + 1)
    height = [years.count(_) for _ in x]

    fig, ax = plt.subplots(1, 1, tight_layout=True)
    ax.bar(x, height)
    ax.set_xticks(x)
    ax.set_xticklabels([str(_) for _ in x])

    ax.set_yticks(np.array([2, 4, 6, 8]) * 1e6)
    ax.set_yticklabels(["2M", "4M", "6M", "8M"])
    ax.set_xlabel("Year")
    ax.set_ylabel("Num. Articles")
    fig.show()

    return fig
