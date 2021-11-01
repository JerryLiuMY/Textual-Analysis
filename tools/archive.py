import mysql.connector
from global_settings import user, host, password
from global_settings import DATA_PATH
import mysql.connector
import os
import numpy as np


def create_stkcd_all():
    csmar = mysql.connector.connect(user=user, password=password, host=host, database="CSMAR")
    csmar_cursor = csmar.cursor()
    csmar_cursor.execute(f"""SELECT Stkcd FROM TRD_Dalyr""")
    stkcd_query = csmar_cursor.fetchall()
    stkcd_all = sorted(set([_[0] for _ in stkcd_query]))

    with open(os.path.join(DATA_PATH, "stkcd_all.npy"), "wb") as f:
        np.save(f, stkcd_all)


def create_dalym():
    csmar = mysql.connector.connect(user=user, password=password, host=host, database="CSMAR")
    csmar_cursor = csmar.cursor()
    csmar_cursor.execute(f"""SELECT * FROM TRD_Dalym""")
    dalym = csmar_cursor.fetchall()

    with open(os.path.join(DATA_PATH, "dalym.csv"), "wb") as f:
        np.save(f, dalym)
