import pickle
import numpy as np
import pytz
import math
import datetime
from global_setting import CRSP_FOLDER, SP500_TABLE, LINK_TABLE
from global_setting import SEARCH_DAY_BEFORE, SEARCH_DAY_AFTER, APPEND_DAY_BEFORE, APPEND_DAY_AFTER, PRICE_LIST_LEN


def get_price_cap(dsf_df, permno, date, oc):
    try:
        sid = dsf_df.loc[permno, date]['SID']
        if sid[0] == '6':
            return np.nan, np.nan

        shrout = float(dsf_df.loc[permno, date]['SHROUT'])
        if oc == 'open':
            open_price = abs(float(dsf_df.loc[permno, date]['OPENPRC']))
            open_cap = abs(open_price) * shrout
            return open_price, open_cap
        else:
            close_price = abs(float(dsf_df.loc[permno, date]['PRC']))
            close_cap = abs(close_price) * shrout
            return close_price, close_cap
    except:
        return np.nan, np.nan


def get_price_cap_sp500_list(dsf_df, est_date, est_time, permno, oc):
    # Note: splitting the dsf into year may cause the open/price at the beginning or the end of the year being zero
    price_before_list = []
    price_after_list = []
    cap_list = []
    sp500_list = []

    hour = est_time.split(':')[0]
    minute = est_time.split(':')[1]
    if oc == 'open' and int(hour) == 9 and int(minute) <= 30:
        return np.nan, np.nan, np.nan

    if oc == 'open':
        if int(hour) < 9:
            date_before_list = -np.arange(SEARCH_DAY_BEFORE) - 1
            date_after_list = np.arange(SEARCH_DAY_AFTER)
        else:
            date_before_list = -np.arange(SEARCH_DAY_BEFORE)
            date_after_list = np.arange(SEARCH_DAY_AFTER) + 1

    else:
        if int(hour) < 16:
            date_before_list = -np.arange(SEARCH_DAY_BEFORE) - 1
            date_after_list = np.arange(SEARCH_DAY_AFTER)
        else:
            date_before_list = -np.arange(SEARCH_DAY_BEFORE)
            date_after_list = np.arange(SEARCH_DAY_AFTER) + 1

    for delta_day in date_before_list:
        if len(price_before_list) < APPEND_DAY_BEFORE:
            new_date = get_date(est_date, int(delta_day))
            price, cap = get_price_cap(dsf_df, permno, new_date, oc)

            if not math.isnan(price):
                price_before_list.append(price)
                sp500 = get_sp500(new_date, oc)
                cap_list.append(cap)
                sp500_list.append(sp500)

    for delta_day in date_after_list:
        if len(price_after_list) < APPEND_DAY_AFTER:
            new_date = get_date(est_date, int(delta_day))
            price, cap = get_price_cap(dsf_df, permno, new_date, oc)

            if not math.isnan(price):
                price_after_list.append(price)
                sp500 = get_sp500(new_date, oc)
                cap_list.append(cap)
                sp500_list.append(sp500)

    price_list = price_before_list + price_after_list

    if not (len(price_list) == PRICE_LIST_LEN and price_list[3] > 3.0):
        price_list, cap_list, sp500_list = np.nan, np.nan, np.nan

    return price_list, cap_list, sp500_list
