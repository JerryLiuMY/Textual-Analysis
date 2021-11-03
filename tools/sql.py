import numpy as np


def query_dalyr(cursor, stkcd, date, select="all"):
    """ Fetch characteristics of a stock on a particular date
    :param cursor: csmar mysql.cursor object
    :param stkcd: company code
    :param date: date of query
    :param select: the variables to query
    :return: a list of selected variables
    """

    if select not in ["all", "MARKETTYPE", "CLSPRC", "DSMVOSD", "DRETND"]:
        raise ValueError("Invalid variable name")

    variables = "MARKETTYPE, CLSPRC, DSMVOSD, DRETND" if select == "all" else select

    cursor.execute(f"""
        SELECT {variables}
        FROM TRD_Dalyr
        WHERE Stkcd = '{stkcd}' AND Trddt = '{date}'
        """)

    output = cursor.fetchall()
    result = [np.nan] * (variables.count(",") + 1) if len(output) == 0 else list(output[0])

    return result
