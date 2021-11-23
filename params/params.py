import os
import pandas as pd
from global_settings import DATA_PATH

# https://github.com/MengLingchao/Chinese_financial_sentiment_dictionary
xlsx_dict = pd.ExcelFile(os.path.join(DATA_PATH, "Chinese_Dict.xlsx"))
pos_dict = [_.strip() for _ in xlsx_dict.parse("positive").iloc[:, 0]]
neg_dict = [_.strip() for _ in xlsx_dict.parse("negative").iloc[:, 0]]
full_dict = pos_dict + neg_dict

date0_min = "2015-01-07"
date0_max = "2019-07-30"

params_dict = {
    "ssestm": {"pen": 0.01}
}
