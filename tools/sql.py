import mysql.connector
from global_settings import user, host, password


def query_csmar(stkcd, date):
    csmar = mysql.connector.connect(user=user, password=password, host=host, database="CSMAR")
    csmar_cursor = csmar.cursor()
    csmar_cursor.execute(f"""
        SELECT DRETND, DSMVOSD
        FROM TRD_Dalyr
        WHERE Stkcd = '{stkcd}' AND Trddt = '{date}'
        """)

    (ret, mkt_cap) = csmar_cursor.fetchall()

    return ret, mkt_cap
