import os
import pandas as pd
from global_settings import DATA_PATH, CLEAN_PATH


def split_data(clean_file, split_num):
    """ Split the cleaned data
    :param clean_file: name of the cleaned file
    :param split_num: number of files to generate
    """

    # load cleaned file
    print(f"Loading {clean_file}...")
    clean_df = pd.read_csv(os.path.join(DATA_PATH, clean_file))
    size = clean_df.shape[0]
    sub_size = int(size / split_num)

    for idx, iloc in enumerate(range(0, size, sub_size)):
        sub_df_clean = clean_df.iloc[iloc: iloc + sub_size, :].reset_index(inplace=False, drop=True)
        sub_file_clean = clean_file.split(".")[0] + f"_{str(idx).zfill(3)}.csv"

        print(f"Saving to {sub_file_clean}...")
        sub_df_clean.to_csv(os.path.join(CLEAN_PATH, sub_file_clean), index=False)
