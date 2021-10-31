import pickle
import numpy as np
import pytz
import os
import math
import datetime
import pandas as pd
from global_setting import CRSP_FOLDER, SP500_TABLE, LINK_TABLE
from global_setting import SEARCH_DAY_BEFORE, SEARCH_DAY_AFTER, APPEND_DAY_BEFORE, APPEND_DAY_AFTER, PRICE_LIST_LEN


def load_dsf(year):
    dsf_df = pd.DataFrame(pd.read_hdf(os.path.join(CRSP_FOLDER, f'dsf_{year}.h5'), key='df'))
    dsf_df.set_index(['PERMNO', 'DATE'], inplace=True)
    print(f'{datetime.datetime.now()} dsf_{year}_df is loaded!')

    return dsf_df


# ------------------------------ Helper Functions ------------------------------
def get_date(old_date, delta_day):
    """
    :param str old_date: the old date in 'YYYYMMDD' format
    :param int delta_day: the difference between old day and new day
    :return str new_day: the new date in 'YYYYMMDD' format
    """
    year = int(old_date.split('-')[0].lstrip('0'))
    month = int(old_date.split('-')[1].lstrip('0'))
    day = int(old_date.split('-')[2].lstrip('0'))
    old_date = datetime.date(year, month, day)
    delta_date = datetime.timedelta(days=delta_day)
    new_date = old_date + delta_date

    new_date = str(new_date.year).rjust(4, '0') + '-' + str(new_date.month).rjust(2, '0') + '-' + str(new_date.day).rjust(2, '0')

    return new_date


def get_permno(ticker, date):
    try:
        ticker_df = LINK_TABLE.loc[[ticker], :]
        for index, row in ticker_df.iterrows():
            namedt = int(row['NAMEDT'])
            nameenddt = int(row['NAMEENDT'])
            if namedt < int(date) < nameenddt:  # TODO: Check if the date is inclusive or exclusive
                permno = row['PERMNO']
                return permno
    except:
        return None


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


def get_sp500(date, oc):
    """
    :param str day: day in 'YYYYMMDD' format
    :return float sp500: the sp500 index of the company on the day
    """
    if oc == 'open':
        sp500_open = float(SP500_TABLE.loc[[date], ['Open']].values)
        return sp500_open
    else:
        sp500_close = float(SP500_TABLE.loc[[date], ['Close']].values)
        return sp500_close


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


# ------------------------------ Query Data ------------------------------
def filter_news(news_df):
    # Filter news with exactly one ticker
    company_list = np.array(news_df['Company'])
    title_list = np.array(news_df['Title'])
    article_list = np.array(news_df['Article'])
    index_list = []
    for i in range(len(company_list)):
        if (company_list[i] is not None) and (len(company_list[i]) == 1):
            content_list = np.append(title_list[i], article_list[i])
            if len(content_list) != 0:
                index_list.append(i)
    news_df = news_df.iloc[index_list, :]
    news_df = news_df.reset_index(inplace=False, drop=True)

    return news_df


def append_est_date_time(news_df):
    company_list = np.array(news_df['Company'])
    GMT_time_list = np.array(news_df['System_GMT_Time'])
    est_date_list = []
    est_time_list = []
    for i in range(len(company_list)):
        gmt_year = int(GMT_time_list[i][:4])
        gmt_month = int(GMT_time_list[i][4:6])
        gmt_day = int(GMT_time_list[i][6:8])
        gmt_hour = int(GMT_time_list[i][9:11])
        gmt_minute = int(GMT_time_list[i][11:13])
        gmt_datetime = datetime.datetime(gmt_year, gmt_month, gmt_day, gmt_hour, gmt_minute)
        gmt = pytz.timezone('GMT')
        eastern = pytz.timezone('US/Eastern')
        gmt_datetime = gmt.localize(gmt_datetime)
        est_datetime = gmt_datetime.astimezone(eastern)

        est_date = str(est_datetime.year).zfill(4) + '-' + str(est_datetime.month).zfill(2) + '-' + str(est_datetime.day).zfill(2)
        est_time = str(est_datetime.hour).zfill(2) + ':' + str(est_datetime.minute).zfill(2)

        est_date_list.append(est_date)
        est_time_list.append(est_time)

    news_df['est_date'] = est_date_list
    news_df['est_time'] = est_time_list

    return news_df


def filter_append_permno(news_df):
    company_list = np.array(news_df['Company'])
    est_date_list = np.array(news_df['est_date'])
    index_list = []
    permno_list = []
    for i in range(len(company_list)):
        company = company_list[i][0]
        est_date = est_date_list[i].replace('-', '')
        permno = get_permno(company, est_date)
        if permno is not None:
            index_list.append(i)
            permno_list.append(permno)
    news_df = news_df.iloc[index_list, :]
    news_df = news_df.reset_index(inplace=False, drop=True)
    news_df['permno'] = permno_list

    return news_df


def append_price_cap_sp500_return(news_df, dsf_df):
    company_list = np.array(news_df['Company'])
    permno_list = np.array(news_df['permno'])
    est_date_list = np.array(news_df['est_date'])
    est_time_list = np.array(news_df['est_time'])
    open_price_matrix = []
    open_cap_matrix = []
    open_sp500_matrix = []
    open_price_return_matrix = []
    open_sp500_return_matrix = []
    close_price_matrix = []
    close_cap_matrix = []
    close_sp500_matrix = []
    close_price_return_matrix = []
    close_sp500_return_matrix = []

    for i in range(len(company_list)):
        if i % 1000 == 0:
            print(f'{datetime.datetime.now()} Fetched Price for {i} Companies!')
        permno = permno_list[i]
        est_time = est_time_list[i]
        est_date = est_date_list[i]
        open_price_list, open_cap_list, open_sp500_list = get_price_cap_sp500_list(dsf_df, est_date, est_time, permno, oc='open')
        close_price_list, close_cap_list, close_sp500_list = get_price_cap_sp500_list(dsf_df, est_date, est_time, permno, oc='close')

        if (not np.isnan(open_price_list).any()) and (not np.isnan(open_sp500_list).any()) and (not np.isnan(open_cap_list).any()):
            open_price_return_list = np.diff(open_price_list) / open_price_list[:-1]
            open_sp500_return_list = np.diff(open_sp500_list) / open_sp500_list[:-1]
            open_price_matrix.append(open_price_list)
            open_cap_matrix.append(open_cap_list)
            open_sp500_matrix.append(open_sp500_list)
            open_price_return_matrix.append(open_price_return_list)
            open_sp500_return_matrix.append(open_sp500_return_list)
        else:
            open_price_matrix.append(np.nan)
            open_cap_matrix.append(np.nan)
            open_sp500_matrix.append(np.nan)
            open_price_return_matrix.append(np.nan)
            open_sp500_return_matrix.append(np.nan)

        if (not np.isnan(close_price_list).any()) and (not np.isnan(close_sp500_list).any()) and (not np.isnan(close_cap_list).any()):
            close_price_return_list = np.diff(close_price_list) / close_price_list[:-1]
            close_sp500_return_list = np.diff(close_sp500_list) / close_sp500_list[:-1]
            close_price_matrix.append(close_price_list)
            close_cap_matrix.append(close_cap_list)
            close_sp500_matrix.append(close_sp500_list)
            close_price_return_matrix.append(close_price_return_list)
            close_sp500_return_matrix.append(close_sp500_return_list)
        else:
            close_price_matrix.append(np.nan)
            close_cap_matrix.append(np.nan)
            close_sp500_matrix.append(np.nan)
            close_price_return_matrix.append(np.nan)
            close_sp500_return_matrix.append(np.nan)

        # print(f'open price {open_price_list}')
        # print(f'open capitalization {open_cap_list}')
        # print(f'open sp500 {open_sp500_list}')
        # print(f'close price {close_price_list}')
        # print(f'close capitalization {close_cap_list}')
        # print(f'close sp500 {close_sp500_list}')

    news_df['open_price'] = open_price_matrix
    news_df['open_cap'] = open_cap_matrix
    news_df['open_sp500'] = open_sp500_matrix
    news_df['open_price_return'] = open_price_return_matrix
    news_df['open_sp500_return'] = open_sp500_return_matrix
    news_df['close_price'] = close_price_matrix
    news_df['close_cap'] = close_cap_matrix
    news_df['close_sp500'] = close_sp500_matrix
    news_df['close_price_return'] = close_price_return_matrix
    news_df['close_sp500_return'] = close_sp500_return_matrix

    # 1 2 3 | 4 5 6 7 8
    # 1-2 2-3 | 3-4 4-5 5-6 6-7 7-8

    return news_df


if __name__ == '__main__':
    raw_path = '/Volumes/Seagate_2T/news_data/2010-12_cleaned.pkl'
    date = str(raw_path).split('/')[-1].split('.')[0].split('_')[0]
    year = date.split('-')[0]
    month = date.split('-')[1]
    dsf_df = load_dsf(year)

    with open(str(raw_path), 'rb') as handle:
        news_df = pickle.load(handle)
    news_df = filter_news(news_df)
    news_df = append_est_date_time(news_df)
    news_df = filter_append_permno(news_df)
    news_df = append_price_cap_sp500_return(news_df, dsf_df)
