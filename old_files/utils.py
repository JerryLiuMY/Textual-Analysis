import pandas as pd 
import os
import pickle
import numpy as np
import scipy.stats as st
from global_setting import TEMP_FOLDER
from global_setting import SP500_TABLE, LOOK_UP_FOLDER, LM_DICTIONARY, FULL_DICT_SIZE
from global_setting import LM_SCORE_FOLDER, DCX_SCORE_FOLDER, MIXED_SCORE_FOLDER
from global_setting import LM_PARAMS_FOLDER, DCX_PARAMS_FOLDER, MIXED_PARAMS_FOLDER
from global_setting import LM_RETURN_FOLDER, DCX_RETURN_FOLDER, MIXED_RETURN_FOLDER
from global_setting import LM_ANALYSIS_FOLDER, DCX_ANALYSIS_FOLDER, MIXED_ANALYSIS_FOLDER
from global_setting import TRAIN_LEN, TEST_LEN, CROSS_LEN
import datetime


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


def get_year_train(year_test_i, year_test_f):
    year_train_i = year_test_i - CROSS_LEN - TRAIN_LEN
    year_train_f = year_test_f - TEST_LEN - CROSS_LEN

    return year_train_i, year_train_f


def get_params_folder(dict_type):
    if dict_type == 'lm':
        PARAMS_FOLDER = LM_PARAMS_FOLDER
    elif dict_type == 'dcx':
        PARAMS_FOLDER = DCX_PARAMS_FOLDER
    else:
        PARAMS_FOLDER = MIXED_PARAMS_FOLDER

    return PARAMS_FOLDER


def get_score_folder(dict_type):
    if dict_type == 'lm':
        SCORE_FOLDER = LM_SCORE_FOLDER
    elif dict_type == 'dcx':
        SCORE_FOLDER = DCX_SCORE_FOLDER
    else:
        SCORE_FOLDER = MIXED_SCORE_FOLDER

    return SCORE_FOLDER


def get_return_folder(dict_type):
    if dict_type == 'lm':
        RETURN_FOLDER = LM_RETURN_FOLDER
    elif dict_type == 'dcx':
        RETURN_FOLDER = DCX_RETURN_FOLDER
    else:
        RETURN_FOLDER = MIXED_RETURN_FOLDER

    return RETURN_FOLDER


def get_analysis_folder(dict_type):
    if dict_type == 'lm':
        ANALYSIS_FOLDER = LM_ANALYSIS_FOLDER
    elif dict_type == 'dcx':
        ANALYSIS_FOLDER = DCX_ANALYSIS_FOLDER
    else:
        ANALYSIS_FOLDER = MIXED_ANALYSIS_FOLDER

    return ANALYSIS_FOLDER


def build_dictionary(year_train_i, year_train_f, dict_type):
    if dict_type == 'lm':
        dictionary = LM_DICTIONARY
    elif dict_type == 'dcx':
        dcx_dictionary_file = os.path.join(LOOK_UP_FOLDER, str(year_train_i) + '_' + str(year_train_f) + '.xlsx')
        dcx_dictionary = list(pd.read_excel(dcx_dictionary_file, 'Full', header=None, dtype=str).iloc[:, 0])
        dictionary = dcx_dictionary
    else:
        dcx_dictionary_file = os.path.join(LOOK_UP_FOLDER, str(year_train_i) + '_' + str(year_train_f) + '.xlsx')
        dcx_dictionary = list(pd.read_excel(dcx_dictionary_file, 'Full', header=None, dtype=str).iloc[:, 0])
        dictionary = sorted(list(set(LM_DICTIONARY + dcx_dictionary)))

    return dictionary


def combine_word_occur_return(year_i, year_f, oc='open'):
    word_occur_return_df_list = []
    for year in range(year_i, year_f + 1):
        max_month = 7 if year == 2017 else 12
        for month in range(1, max_month+1):
            temp_path = os.path.join(TEMP_FOLDER, str(year) + '_' + str(month).zfill(2) + '.pkl')
            with open(temp_path, 'rb') as handle:
                content = pickle.load(handle)
                word_occur_return_df_list.append(content)
    word_occur_return_df_combined = pd.concat(word_occur_return_df_list, axis=0)

    if oc == 'open':
        word_occur_return_df_combined.dropna(inplace=True,
                                             subset=['worthy', 'open_price', 'open_cap', 'open_sp500',
                                                     'open_price_return', 'open_sp500_return'])
    else:
        word_occur_return_df_combined.dropna(inplace=True,
                                             subset=['worthy', 'close_price', 'close_cap', 'close_sp500',
                                                     'close_price_return', 'close_sp500_return'])
    return word_occur_return_df_combined


def combine_datetime(word_occur_return_df_combined, year_i, year_f):
    business_date_list_full = list(SP500_TABLE.index)
    business_date_list = [business_day for business_day in business_date_list_full
                          if year_i <= int(business_day.split('-')[0]) <= year_f + 1]
    est_date_list = list(word_occur_return_df_combined['est_date'])
    est_time_list = list(word_occur_return_df_combined['est_time'])
    est_date_list_reassigned = []
    est_time_list_reassigned = []
    for i in range(len(est_date_list)):
        est_date = est_date_list[i]
        est_time = est_time_list[i]
        hour = est_time.split(':')[0]

        # If the date is a business day and hour before 9:00, keep the datetime unchanged, else find the next business
        # day and set the time to 5:00am of that day
        if (est_date in business_date_list) and (int(hour) < 9):
            est_date_list_reassigned.append(est_date)
            est_time_list_reassigned.append(est_time)
        else:
            delta_day = 1
            est_date_delta = get_date(est_date, delta_day)
            while est_date_delta not in business_date_list:
                delta_day += 1
                est_date_delta = get_date(est_date, delta_day)
            est_date_list_reassigned.append(est_date_delta)
            est_time_list_reassigned.append('05:00')

    word_occur_return_df_combined['est_date'] = est_date_list_reassigned
    word_occur_return_df_combined['est_time'] = est_time_list_reassigned

    return word_occur_return_df_combined


def combine_permno(word_occur_return_df_day, dictionary):
    unique_permno_list = np.unique(word_occur_return_df_day.index)
    word_occur_return_matrix = []
    for unique_permno in unique_permno_list:
        word_occur_array = word_occur_return_df_day.loc[[unique_permno], :].loc[:, dictionary].values.sum(axis=0)
        return_array = word_occur_return_df_day.loc[[unique_permno], :].iloc[:, FULL_DICT_SIZE:].values[0]
        word_occur_return_array = np.append(word_occur_array, return_array)
        word_occur_return_matrix.append(word_occur_return_array)

    word_occur_return_df_day = pd.DataFrame(word_occur_return_matrix, index=unique_permno_list,
                                            columns=dictionary + list(word_occur_return_df_day.columns)[FULL_DICT_SIZE:])

    return word_occur_return_df_day


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), st.sem(a)
    h = se * st.t.ppf((1 + confidence) / 2., n-1)
    return m, [m-h, m+h]

