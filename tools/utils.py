import datetime
from dateutil import tz
import os
import numpy as np
import pandas as pd
from global_settings import DATA_PATH


def convert_datetime(timestamp):
    """ Epoch & Unix Timestamp
    :param timestamp: Epoch & Unix Timestamp
    :return: China date & time Timestamps
    """

    fmt_date = "%Y-%m-%d"
    fmt_time = "%H:%M:%S"

    from_zone = tz.gettz("UTC")
    to_zone = tz.gettz("Asia/Shanghai")
    utc_datetime = datetime.datetime.utcfromtimestamp(float(timestamp) / 1000.)
    utc_datetime = utc_datetime.replace(tzinfo=from_zone)
    chn_datetime = utc_datetime.astimezone(to_zone)
    chn_date = chn_datetime.strftime(fmt_date)
    chn_time = chn_datetime.strftime(fmt_time)

    return chn_date, chn_time


def shift_date(date, delta_day):
    """
    :param date: date string in the format of "%Y-%m-%d"
    :param delta_day: number of days to lag
    :return: lagged date in the format of "%Y-%m-%d"
    """
    date_fmt = "%Y-%m-%d"
    date_datetime = datetime.datetime.strptime(date, date_fmt)
    new_date_datetime = date_datetime + datetime.timedelta(days=delta_day)
    new_date = datetime.datetime.strftime(new_date_datetime, date_fmt)

    return new_date


def match_date(dalym_file):
    os.path.join(DATA_PATH)

    trddt_all = np.array(sorted(set(dalym["Trddt"])))

    pass
