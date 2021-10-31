import datetime
from datetime import datetime
from dateutil import tz


def convert_datetime(timestamp):
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


def lag_date(date, delta_day):
    """
    :param date: date string in the format of "%Y-%m-%d"
    :param delta_day: number of days to lag
    :return: lagged date in the format of "%Y-%m-%d"
    """
    datetime.datetime.strptime(date, "%Y-%m-%d")
    delta_date = datetime.timedelta(days=delta_day)
    new_date = date + delta_date

    return new_date
