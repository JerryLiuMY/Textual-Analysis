import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dateutil.relativedelta import relativedelta
from datetime import datetime
plt.style.use('ggplot')


def plot_backtest(ret_df, dalym):
    """ plot cumulative return from backtesting
    :param ret_df: dataframe of equal weighted returns
    :param dalym: dataframe of index
    :return:
    """

    # equal weighted returns
    ret_le = np.array(ret_df["ret_le"])
    ret_se = np.array(ret_df["ret_se"])
    ret_e = np.array(ret_df["ret_e"])
    cum_le = np.log(np.cumprod(ret_le + 1))
    cum_se = np.log(np.cumprod(ret_se + 1))
    cum_e = np.log(np.cumprod(ret_e + 1))

    # value weighted returns
    ret_lv = np.array(ret_df["ret_lv"])
    ret_sv = np.array(ret_df["ret_sv"])
    ret_v = np.array(ret_df["ret_v"])
    cum_lv = np.log(np.cumprod(ret_lv + 1))
    cum_sv = np.log(np.cumprod(ret_sv + 1))
    cum_v = np.log(np.cumprod(ret_v + 1))

    # index returns
    dalym = dalym.loc[dalym["Trddt"].apply(lambda _: _ in ret_df.index), :]
    mkt_ret = dalym.groupby(by=["Trddt"]).apply(lambda _: np.average(_["Dretmdos"], weights=_["Dnvaltrdtl"], axis=0))
    mkt_cum = np.log(np.cumprod(mkt_ret + 1))

    # xticks and xticklabels
    tick_lab_beg = datetime.strptime(ret_df.index[0], "%Y-%m-%d")
    tick_lab_end = datetime.strptime(ret_df.index[-1], "%Y-%m-%d")
    tick_lab_cur = tick_lab_beg
    tick_labs = []
    while tick_lab_cur <= tick_lab_end:
        tick_labs.append(tick_lab_cur)
        trddt_Ym = (tick_lab_cur + relativedelta(months=6)).strftime("%Y-%m")
        def match_Ym(series): return datetime.strptime(series, "%Y-%m-%d").strftime("%Y-%m") == trddt_Ym
        if len(ret_df.index[pd.Series(ret_df.index).apply(match_Ym)]) == 0:
            break
        else:
            tick_lab_cur = datetime.strptime(ret_df.index[pd.Series(ret_df.index).apply(match_Ym)][0], "%Y-%m-%d")

    def lab_to_tick(tick_lab): return np.where(pd.Series(ret_df.index).apply(lambda _: _ == tick_lab))[0][0]
    tick_labs = [_.strftime("%Y-%m-%d") for _ in tick_labs]
    ticks = [lab_to_tick(tick_lab) for tick_lab in tick_labs]

    # plot cumulative return
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labs)
    ax.grid("on")
    ax.plot(cum_e, "k-")
    ax.plot(cum_le, "b-")
    ax.plot(-cum_se, "r-")
    ax.plot(cum_v, "k--")
    ax.plot(cum_lv, "b--")
    ax.plot(-cum_sv, "r--")
    ax.plot(mkt_cum, 'y-')
    ax.legend(["L-S EW", "L EW", "S EW", "L-S VW", "L VW", "S VW", "Index"])

    return fig
