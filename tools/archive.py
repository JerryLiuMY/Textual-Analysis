import mysql.connector
from global_settings import user, host, password
from global_settings import DATA_PATH
from global_settings import RICH_PATH
import mysql.connector
from datetime import datetime
import numpy as np
import pandas as pd
import glob
import os


def create_stkcd_all():
    """Fetch all stkcd company codes from TRD_Dalyr"""
    csmar = mysql.connector.connect(user=user, password=password, host=host, database="CSMAR")
    csmar_cursor = csmar.cursor()
    csmar_cursor.execute(f"""SELECT Stkcd FROM TRD_Dalyr""")
    stkcd_query = csmar_cursor.fetchall()
    stkcd_all = sorted(set([_[0] for _ in stkcd_query]))
    np.save(os.path.join(DATA_PATH, "stkcd_all.npy"), stkcd_all)


def create_dalym():
    """Fetch the TRD_Dalym dataframe"""
    csmar = mysql.connector.connect(user=user, password=password, host=host, database="CSMAR")
    csmar_cursor = csmar.cursor()
    csmar_cursor.execute(f"""SELECT * FROM TRD_Dalym""")
    dalym = csmar_cursor.fetchall()
    dalym.to_csv(os.path.join(DATA_PATH, "dalym.csv"))


def get_date0_range():
    """Fetch the range of date0 from the enriched dataframe"""
    sub_file_rich_li = sorted([_.split("/")[-1] for _ in glob.glob(os.path.join(RICH_PATH, "*"))])

    df_rich = pd.DataFrame()
    for sub_file_rich in sorted(sub_file_rich_li):
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Loading {sub_file_rich}")
        sub_df_rich = pd.read_csv(os.path.join(RICH_PATH, sub_file_rich))
        df_rich = df_rich.append(sub_df_rich)

    df_rich.reset_index(inplace=True, drop=True)
    date0_all = sorted(set(df_rich["date_0"]))

    return min(date0_all), max(date0_all)
