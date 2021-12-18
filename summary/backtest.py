import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
from global_settings import OUTPUT_PATH
plt.style.use("ggplot")


def backtest(model_name, dalym):
    """ plot cumulative return from backtesting
    :param model_name: model name
    :param dalym: dataframe of index
    """

    # define model path
    model_path = os.path.join(OUTPUT_PATH, model_name)
    ret_df = pd.read_csv(os.path.join(model_path, "ret_df.csv"), index_col=0)

    # equal weighted returns
    ret_le = np.array(ret_df["ret_le"])
    ret_se = -np.array(ret_df["ret_se"])
    ret_e = np.array(ret_df["ret_e"])
    cum_le = np.log(np.cumprod(ret_le + 1))
    cum_se = np.log(np.cumprod(ret_se + 1))
    cum_e = np.log(np.cumprod(ret_e + 1))

    # value weighted returns
    ret_lv = np.array(ret_df["ret_lv"])
    ret_sv = -np.array(ret_df["ret_sv"])
    ret_v = np.array(ret_df["ret_v"])
    cum_lv = np.log(np.cumprod(ret_lv + 1))
    cum_sv = np.log(np.cumprod(ret_sv + 1))
    cum_v = np.log(np.cumprod(ret_v + 1))

    # index returns
    dalym = dalym.loc[dalym["Trddt"].apply(lambda _: _ in ret_df.index), :]
    mkt_ret = dalym.groupby(by=["Trddt"]).apply(lambda _: np.average(_["Dretmdos"], weights=_["Dnvaltrdtl"], axis=0))
    mkt_cum = np.log(np.cumprod(mkt_ret + 1))

    # compute average returns & sharpe ratios
    ave_le, ave_se, ave_e = map(get_ave, [cum_le, cum_se, cum_e])
    ave_lv, ave_sv, ave_v = map(get_ave, [cum_lv, cum_sv, cum_v])
    sha_le, sha_se, sha_e = map(get_sha, [ret_le, ret_se, ret_e])
    sha_lv, sha_sv, sha_v = map(get_sha, [ret_lv, ret_sv, ret_v])
    ave_idx, sha_idx = get_ave(mkt_cum), get_sha(mkt_ret)
    summary = {
        "ave_le": ave_le, "ave_se": ave_se, "ave_e": ave_e,
        "ave_lv": ave_lv, "ave_sv": ave_sv, "ave_v": ave_v,
        "sha_le": sha_le, "sha_se": sha_se, "sha_e": sha_e,
        "sha_lv": sha_lv, "sha_sv": sha_sv, "sha_v": sha_v,
        "ave_idx": ave_idx, "sha_idx": sha_idx
    }

    with open(os.path.join(model_path, f"summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # plot cumulative return
    xticks, xlabs = get_xticklabs(ret_df)
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabs)
    ax.grid("on")
    ax.plot(cum_e, "k-")
    ax.plot(cum_le, "b-")
    ax.plot(-cum_se, "r-")
    ax.plot(cum_v, "k--")
    ax.plot(cum_lv, "b--")
    ax.plot(-cum_sv, "r--")
    ax.plot(mkt_cum, 'y-')
    ax.legend(["L-S EW", "L EW", "S EW", "L-S VW", "L VW", "S VW", "Index"])
    ax.set_xlabel("Dates")
    ax.set_ylabel("log(cum_ret)")

    fig.savefig(os.path.join(model_path, "backtest.pdf"), bbox_inches="tight")


def get_ave(cum):
    """ get average daily return from array of cumulative returns
    :param cum: array of cumulative returns
    """

    ave = (np.exp(cum[-1]) - 1) / (len(cum) - 1)

    return ave


def get_sha(ret):
    """ get sharpe ratio from array of returns
    :param ret: array of daily returns
    """

    sha = (np.mean(ret) / np.std(ret)) * np.sqrt(252)

    return sha


def get_xticklabs(ret_df):
    """ get xticks and xlabs for cumulative returns of backtesting
    :param ret_df: return dataframe over the rolling period
    """

    # define tick labels
    lab_beg = datetime.strptime(ret_df.index[0], "%Y-%m-%d")
    lab_end = datetime.strptime(ret_df.index[-1], "%Y-%m-%d")
    lab_cur = lab_beg
    xlabs = []

    while lab_cur <= lab_end:
        xlabs.append(lab_cur)
        trddt_Ym = (lab_cur + relativedelta(months=6)).strftime("%Y-%m")
        def match_Ym(series): return datetime.strptime(series, "%Y-%m-%d").strftime("%Y-%m") == trddt_Ym
        matched_dates = ret_df.index[pd.Series(ret_df.index).apply(match_Ym)]
        if len(matched_dates) != 0:
            lab_cur = datetime.strptime(matched_dates[0], "%Y-%m-%d")
        else:
            break

    # define ticks
    def lab_to_tick(ticklab): return np.where(pd.Series(ret_df.index).apply(lambda _: _ == ticklab))[0][0]
    xlabs = [ticklab.strftime("%Y-%m-%d") for ticklab in xlabs]
    xticks = [lab_to_tick(ticklab) for ticklab in xlabs]

    return xticks, xlabs
