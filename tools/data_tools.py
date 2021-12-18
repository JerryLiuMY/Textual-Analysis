import json
import os
from dateutil import tz
from datetime import datetime, timedelta
from global_settings import trddt_all
from global_settings import LOG_PATH


def init_data_log():
    """Initialize the log file for step-by-step details for our sample filters."""

    data_log = {
        "original": 0,
        "available": 0,
        "drop_nan": 0,
        "single_tag": 0,
        "match_stkcd": 0
    }

    print("data_log.json initialized")
    with open(os.path.join(LOG_PATH, "data_log.json"), "w") as f:
        json.dump(data_log, f, indent=2)


def convert_zone(timestamp):
    """ Epoch & Unix Timestamp
    :param timestamp: Epoch & Unix Timestamp
    :return: China date & time Timestamps
    """
    fmt_date = "%Y-%m-%d"
    fmt_time = "%H:%M:%S"

    from_zone = tz.gettz("UTC")
    to_zone = tz.gettz("Asia/Shanghai")
    utc_datetime = datetime.utcfromtimestamp(float(timestamp) / 1000.)
    utc_datetime = utc_datetime.replace(tzinfo=from_zone)
    chn_datetime = utc_datetime.astimezone(to_zone)
    chn_date = chn_datetime.strftime(fmt_date)
    chn_time = chn_datetime.strftime(fmt_time)

    return chn_date, chn_time


def shift_date(date, shift_day):
    """ Shift a date by a delta_day
    :param date: date string in the format of "%Y-%m-%d"
    :param shift_day: number of days to lag
    :return: shifted date in the format of "%Y-%m-%d"
    """
    date_fmt = "%Y-%m-%d"
    date_dt = datetime.strptime(date, date_fmt)
    shifted_date_dt = date_dt + timedelta(days=shift_day)
    shifted_date = datetime.strftime(shifted_date_dt, date_fmt)

    return shifted_date


def match_date(date, match_day=0):
    """ Match a date to the specified trading date
    :param date: date string in the format of "%Y-%m-%d"
    :param match_day: number of days before / after the trading date to match
    :return: matched date in the format of "%Y-%m-%d"
    """
    if match_day >= 0:
        matched_date = trddt_all[trddt_all >= date][match_day]
    else:
        matched_date = trddt_all[trddt_all < date][match_day]

    return matched_date
