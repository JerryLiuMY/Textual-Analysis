import os
import json
import numpy as np
import pandas as pd
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from global_settings import OUTPUT_PATH
plt.style.use("ggplot")


def backtest(model_name, dalym):
    """ plot cumulative return from backtesting
    :param model_name: model name
    :param dalym: dataframe of index
    """

    # define model path
    model_path = os.path.join(OUTPUT_PATH, model_name)
    return_sub_path = os.path.join(model_path, "return")
    ret_csv = pd.concat([pd.read_csv(_, index_col=0) for _ in sorted(glob(os.path.join(return_sub_path, "*.csv")))])
    ret_pkl = pd.concat([pd.read_pickle(_) for _ in sorted(glob(os.path.join(return_sub_path, "*.pkl")))])

    # equal weighted returns
    ret_le = np.array(ret_csv["ret_le"])
    ret_se = np.array(ret_csv["ret_se"])
    ret_e = np.array(ret_csv["ret_e"])
    cum_le = np.log(np.cumprod(ret_le + 1))
    cum_se = np.log(np.cumprod(ret_se + 1))
    cum_e = np.log(np.cumprod(ret_e + 1))

    # value weighted returns
    ret_lv = np.array(ret_csv["ret_lv"])
    ret_sv = np.array(ret_csv["ret_sv"])
    ret_v = np.array(ret_csv["ret_v"])
    cum_lv = np.log(np.cumprod(ret_lv + 1))
    cum_sv = np.log(np.cumprod(ret_sv + 1))
    cum_v = np.log(np.cumprod(ret_v + 1))

    # index returns
    dalym = dalym.loc[dalym["Trddt"].apply(lambda _: _ in ret_csv.index), :]
    mkt_ret = dalym.groupby(by=["Trddt"]).apply(lambda _: np.average(_["Dretmdos"], weights=_["Dnvaltrdtl"], axis=0))
    mkt_cum = np.log(np.cumprod(mkt_ret + 1))

    # compute average returns & sharpe ratios
    ave_le, ave_se = map(get_ave, [cum_le, cum_se])
    ave_lv, ave_sv = map(get_ave, [cum_lv, cum_sv])
    ave_e, ave_v = ave_le + ave_se, ave_lv + ave_sv
    sha_le, sha_se, sha_e = map(get_sha, [ret_le, ret_se, ret_e])
    sha_lv, sha_sv, sha_v = map(get_sha, [ret_lv, ret_sv, ret_v])
    ave_idx, sha_idx = get_ave(mkt_cum), get_sha(mkt_ret)

    # compute turnover
    roll = [[i, i + 1] for i in range(ret_pkl.shape[0] - 1)]
    tov_le = np.array([get_tov(ret_pkl.iloc[r, :], "l", "e") for r in roll]).mean()
    tov_se = np.array([get_tov(ret_pkl.iloc[r, :], "s", "e") for r in roll]).mean()
    tov_lv = np.array([get_tov(ret_pkl.iloc[r, :], "l", "v") for r in roll]).mean()
    tov_sv = np.array([get_tov(ret_pkl.iloc[r, :], "s", "v") for r in roll]).mean()
    tov_e = np.array([get_tov(ret_pkl.iloc[r, :], "", "e") for r in roll]).mean()
    tov_v = np.array([get_tov(ret_pkl.iloc[r, :], "", "v") for r in roll]).mean()

    # exploration
    summary = {
        "ave_le": ave_le, "ave_se": ave_se, "ave_e": ave_e,
        "ave_lv": ave_lv, "ave_sv": ave_sv, "ave_v": ave_v,
        "sha_le": sha_le, "sha_se": sha_se, "sha_e": sha_e,
        "sha_lv": sha_lv, "sha_sv": sha_sv, "sha_v": sha_v,
        "ave_idx": ave_idx, "sha_idx": sha_idx,
        "tov_le": tov_le, "tov_se": tov_se, "tov_e": tov_e,
        "tov_lv": tov_lv, "tov_sv": tov_sv, "tov_v": tov_v
    }

    with open(os.path.join(model_path, f"summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # plot cumulative return
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(7, 1)
    ax1 = fig.add_subplot(gs[0:5, :])
    ax2 = fig.add_subplot(gs[5:7, :])
    xticks, xlabs = get_xticklabs(ret_csv)

    # ax1: backtest
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xlabs)
    ax1.grid("on")
    ax1.plot(cum_e, "k-")
    ax1.plot(cum_le, "b-")
    ax1.plot(-cum_se, "r-")
    ax1.plot(cum_v, "k--")
    ax1.plot(cum_lv, "b--")
    ax1.plot(-cum_sv, "r--")
    ax1.plot(mkt_cum, 'y-')
    ax1.legend(["L-S EW", "L EW", "S EW", "L-S VW", "L VW", "S VW", "Index"])
    ax1.set_xlabel("Dates")
    ax1.set_ylabel("log(cum_ret)")

    # ax2: correlation
    pearson_cor = 0.5 * (ret_csv["cor_e"].to_numpy() + ret_csv["cor_v"].to_numpy())
    ax2.stem(ret_csv.index, pearson_cor, linefmt="#A9A9A9", markerfmt=" ", basefmt=" ")
    ax2.scatter(ret_csv.index, pearson_cor, color="#899499", marker=".")
    ax2.set_xticks(xticks)
    ax2.set_xticklabels(xlabs)
    ax2.set_xlabel("Dates")
    ax2.set_ylabel("Correlation")

    fig.tight_layout()
    fig.savefig(os.path.join(model_path, "backtest.pdf"), bbox_inches="tight")


def get_ave(cum):
    """ get average daily return from array of cumulative returns
    :param cum: array of cumulative returns
    """

    ave = (np.exp(cum[-1]) - 1) / (len(cum) - 1)

    return ave


def get_sha(ret):
    """ get annualized sharpe ratio from array of returns
    :param ret: array of daily returns
    """

    rf = 0
    sha = (np.mean(ret - rf) / np.std(ret - rf)) * np.sqrt(252)

    return sha


def get_tov(ret_pkl, ls, ev):
    """ get turnover from the return dataframe
    :param ret_pkl: return pkl
    :param ls: long/short type
    :param ev: equal/value weighted type
    """

    if ev not in ["e", "v"]:
        raise ValueError("Invalid equal/value weighting type")

    if ls in ["l", "s"]:
        sign = +1 if ls == "l" else -1
        stks_b = ret_pkl.loc[:, "".join(["stks_", ls, ev])].iloc[0]
        stks_a = ret_pkl.loc[:, "".join(["stks_", ls, ev])].iloc[1]
        stks = np.unique(np.concatenate([stks_b, stks_a]))

        idcs_b = np.array([np.where(stks == stk_b)[0][0] for stk_b in stks_b])
        idcs_a = np.array([np.where(stks == stk_a)[0][0] for stk_a in stks_a])

        wgts_b = np.zeros(len(stks))
        rets_b = np.zeros(len(stks))
        wgts_a = np.zeros(len(stks))

        if len(idcs_b) != 0:
            wgts_b[idcs_b] = sign * ret_pkl.loc[:, "".join(["wgts_", ls, ev])].iloc[0]
            rets_b[idcs_b] = ret_pkl.loc[:, "".join(["rets_", ls, ev])].iloc[0]
        if len(idcs_a) != 0:
            wgts_a[idcs_a] = sign * ret_pkl.loc[:, "".join(["wgts_", ls, ev])].iloc[1]

        tov = 0.5 * np.sum(np.abs(wgts_a - wgts_b * (1 + rets_b)))

    elif ls == "":
        stks_b_l = ret_pkl.loc[:, "".join(["stks_", "l", ev])].iloc[0]
        stks_b_s = ret_pkl.loc[:, "".join(["stks_", "s", ev])].iloc[0]
        stks_a_l = ret_pkl.loc[:, "".join(["stks_", "l", ev])].iloc[1]
        stks_a_s = ret_pkl.loc[:, "".join(["stks_", "s", ev])].iloc[1]
        stks = np.unique(np.concatenate([stks_b_l, stks_b_s, stks_a_l, stks_a_s]))

        idcs_b_l = np.array([np.where(stks == stk_b_l)[0][0] for stk_b_l in stks_b_l])
        idcs_b_s = np.array([np.where(stks == stk_b_s)[0][0] for stk_b_s in stks_b_s])
        idcs_a_l = np.array([np.where(stks == stk_a_l)[0][0] for stk_a_l in stks_a_l])
        idcs_a_s = np.array([np.where(stks == stk_a_s)[0][0] for stk_a_s in stks_a_s])

        wgts_b = np.zeros(len(stks))
        rets_b = np.zeros(len(stks))
        wgts_a = np.zeros(len(stks))
        sign_l, sign_s = +0.5, -0.5

        if len(idcs_b_l) != 0:
            wgts_b[idcs_b_l] = sign_l * ret_pkl.loc[:, "".join(["wgts_", "l", ev])].iloc[0]
            rets_b[idcs_b_l] = ret_pkl.loc[:, "".join(["rets_", "l", ev])].iloc[0]
        if len(idcs_b_s) != 0:
            wgts_b[idcs_b_s] = sign_s * ret_pkl.loc[:, "".join(["wgts_", "s", ev])].iloc[0]
            rets_b[idcs_b_s] = ret_pkl.loc[:, "".join(["rets_", "s", ev])].iloc[0]
        if len(idcs_a_l) != 0:
            wgts_a[idcs_a_l] = sign_l * ret_pkl.loc[:, "".join(["wgts_", "l", ev])].iloc[1]
        if len(idcs_a_s) != 0:
            wgts_a[idcs_a_s] = sign_s * ret_pkl.loc[:, "".join(["wgts_", "s", ev])].iloc[1]

        tov = 0.5 * np.sum(np.abs(wgts_a - wgts_b * (rets_b + 1)))

    else:
        raise ValueError("Invalid long/short type")

    return tov


def get_xticklabs(ret_df):
    """ get xticks and xlabs for cumulative returns of backtesting
    :param ret_df: return dataframe over the rolling period
    """

    # define xlabs
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

    # define xticks
    def lab_to_tick(lab): return np.where(pd.Series(ret_df.index).apply(lambda _: _ == lab))[0][0]
    xlabs = [xlab.strftime("%Y-%m-%d") for xlab in xlabs]
    xticks = [lab_to_tick(xlab) for xlab in xlabs]

    return xticks, xlabs
