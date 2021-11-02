import mysql.connector
import numpy as np
from global_settings import user, host, password

csmar = mysql.connector.connect(user=user, password=password, host=host, database="CSMAR")
csmar_cursor = csmar.cursor()


def query_dalyr(stkcd, date, select="all"):
    """ Fetch characteristics of a stock on a particular date
    :param stkcd: company code
    :param date: date of query
    :param select: the variables to query
    :return: a list of selected variables
    """

    if select not in ["all", "MARKETTYPE", "CLSPRC", "DSMVOSD", "DRETND"]:
        raise ValueError("Invalid variable name")

    variables = "MARKETTYPE, CLSPRC, DSMVOSD, DRETND" if select == "all" else select

    csmar_cursor.execute(f"""
        SELECT {variables}
        FROM TRD_Dalyr
        WHERE Stkcd = '{stkcd}' AND Trddt = '{date}'
        """)

    output = csmar_cursor.fetchall()
    result = [np.nan] * (variables.count(",") + 1) if len(output) == 0 else list(output[0])

    return result
