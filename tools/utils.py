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


def shift_date(date, delta_day):
    """ Shift a date by a delta_day
    :param date: date string in the format of "%Y-%m-%d"
    :param delta_day: number of days to lag
    :return: shifted date in the format of "%Y-%m-%d"
    """
    date_fmt = "%Y-%m-%d"
    date_dt = datetime.datetime.strptime(date, date_fmt)
    shifted_date_dt = date_dt + datetime.timedelta(days=delta_day)
    shifted_date = datetime.datetime.strftime(shifted_date_dt, date_fmt)

    return shifted_date


def match_date(date):
    """ Match a date to the next nearest trading date (the date itself inclusive)
    :param date: date string in the format of "%Y-%m-%d"
    :return: matched date in the format of "%Y-%m-%d"
    """
    matched_date = trddt_all[trddt_all >= date][0]

    return matched_date
