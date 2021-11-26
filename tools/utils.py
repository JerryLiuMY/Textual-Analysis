import datetime
from dateutil import tz
from global_settings import trddt_all


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


def shift_date(date, shift_day):
    """ Shift a date by a delta_day
    :param date: date string in the format of "%Y-%m-%d"
    :param shift_day: number of days to lag
    :return: shifted date in the format of "%Y-%m-%d"
    """
    date_fmt = "%Y-%m-%d"
    date_dt = datetime.datetime.strptime(date, date_fmt)
    shifted_date_dt = date_dt + datetime.timedelta(days=shift_day)
    shifted_date = datetime.datetime.strftime(shifted_date_dt, date_fmt)

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


def get_window():


    pass
