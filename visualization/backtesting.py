import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import seaborn as sns
sns.set()
plt.style.use('ggplot')


def plot_backtest(return_df, num_days_cum, year_list):
    """ plot cumulative return from backtesting
    :param return_df: dataframe of returns
    :return:
    """
    return_file_list = []
    # RETURN_FOLDER = get_return_folder(dict_type)
    # for file in glob.glob(os.path.join(RETURN_FOLDER, '*.pkl')):
    #     return_file_list.append(file)
    # return_file_list = sorted(return_file_list)
    #
    # year_list = np.arange(YEAR_TEST_MIN, YEAR_TEST_MAX)
    # year_return_df_list = []
    # year_num_day_list = []
    # for file in return_file_list:
    #     year_return_df = pd.read_pickle(file)
    #     year_num_day = np.shape(year_return_df)[0]
    #     year_return_df_list.append(year_return_df)
    #     year_num_day_list.append(year_num_day)
    # return_df = pd.concat(year_return_df_list, axis=0)
    # num_days_cum = np.cumsum(year_num_day_list)
    # num_days_cum = np.append(1, num_days_cum)

    # calculate cumulative return
    ret_le = np.array(return_df["ret_le"]) + 1
    ret_se = -np.array(return_df["ret_se"]) + 1
    ret_e = ret_le + ret_se - 1
    cum_le = np.log(np.cumprod(ret_le))
    cum_se = np.log(np.cumprod(ret_se))
    cum_e = np.log(np.cumprod(ret_e))

    ret_lv = np.array(return_df["ret_lv"]) + 1
    ret_sv = -np.array(return_df["ret_sv"]) + 1
    ret_v = ret_lv + ret_sv - 1
    cum_lv = np.log(np.cumprod(ret_lv))
    cum_sv = np.log(np.cumprod(ret_sv))
    cum_v = np.log(np.cumprod(ret_v))

    mkt_ret = np.array(return_df["mkt_ret"]) + 1
    index_cum = np.log(np.cumprod(mkt_ret))

    # plot cumulative return
    fig, ax = plt.subplots(1, 1, figsize=(14, 7))
    ax.set_xlim(num_days_cum[0], num_days_cum[-1])
    ax.set_xticks(num_days_cum[0::2], year_list[0::2])
    ax.grid('on')
    ax.plot(cum_e, 'k-')
    ax.plot(cum_le, 'b-')
    ax.plot(-cum_se, 'r-')
    ax.plot(cum_v, 'k--')
    ax.plot(cum_lv, 'b--')
    ax.plot(-cum_sv, 'r--')
    ax.plot(index_cum, 'y-')
    ax.legend(['L-S EW', 'L EW', 'S EW', 'L-S VW', 'L VW', 'S VW', 'SP500'])

    return fig
