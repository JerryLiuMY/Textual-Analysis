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
    # return_file_list = []
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

    # Calculate cumulative return
    long_equal_ret = np.array(return_df["long_equal_ret"]) + 1
    long_equal_cum = np.log(np.cumprod(long_equal_ret))
    short_equal_ret = -np.array(return_df["short_equal_ret"]) + 1
    short_equal_cum = np.log(np.cumprod(short_equal_ret))
    equal_ret = long_equal_ret + short_equal_ret - 1
    equal_cum = np.log(np.cumprod(equal_ret))

    long_value_ret = np.array(return_df["long_value_ret"]) + 1
    long_value_cum = np.log(np.cumprod(long_value_ret))
    short_value_ret = -np.array(return_df["short_value_ret"]) + 1
    short_value_cum = np.log(np.cumprod(short_value_ret))
    value_ret = long_value_ret + short_value_ret - 1
    value_cum = np.log(np.cumprod(value_ret))

    sp500_ret = np.array(return_df["sp500_ret"]) + 1
    sp500_cum = np.log(np.cumprod(sp500_ret))

    # Plot cumulative return
    fig = plt.figure(1, figsize=(14, 7))
    plt.xlim(num_days_cum[0], num_days_cum[-1])
    plt.xticks(num_days_cum[0::2], year_list[0::2])
    plt.grid('on')
    plt.plot(equal_cum, 'k-')
    plt.plot(long_equal_cum, 'b-')
    plt.plot(-short_equal_cum, 'r-')
    plt.plot(value_cum, 'k--')
    plt.plot(long_value_cum, 'b--')
    plt.plot(-short_value_cum, 'r--')
    plt.plot(sp500_cum, 'y-')
    plt.legend(['L-S EW', 'L EW', 'S EW', 'L-S VW', 'L VW', 'S VW', 'SP500'])
    plt.show()

    return fig
