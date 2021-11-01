import mysql.connector
from global_settings import user, host, password


def query_dalyr(stkcd, date):
    """ Fetch characteristics of a stock on a particular date 
    :param stkcd: company code
    :param date: date of query
    :return: a list of [mkt_type, cls, mkt_cap, ret]
    """
    csmar = mysql.connector.connect(user=user, password=password, host=host, database="CSMAR")
    csmar_cursor = csmar.cursor()
    csmar_cursor.execute(f"""
        SELECT MARKETTYPE, CLSPRC, DSMVOSD, DRETND
        FROM TRD_Dalyr
        WHERE Stkcd = '{stkcd}' AND Trddt = '{date}'
        """)

    return csmar_cursor.fetchall()
