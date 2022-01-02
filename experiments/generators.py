from global_settings import trddt_all, DATA_PATH, RICH_PATH
from glob import glob
import itertools
import os


def generate_files(trddt, textual_name):
    """ Build iterator for files
    :param trddt: list of trddt dates
    :param textual_name: name of textual model
    """

    # define paths
    text_path = os.path.join(DATA_PATH, textual_name)
    extension = "npz" if textual_name == "word_sps" else "pkl"
    sub_file_rich_idx = [_.split("/")[-1].split(".")[0] for _ in glob(os.path.join(RICH_PATH, "*.csv"))]
    sub_text_file_idx = [_.split("/")[-1].split(".")[0] for _ in glob(os.path.join(text_path, "*." + extension))]
    sub_file_rich_idx = [_ for _ in sub_file_rich_idx if _ in trddt]
    sub_text_file_idx = [_ for _ in sub_text_file_idx if _ in trddt]

    if sorted(sub_file_rich_idx) != sorted(sub_text_file_idx):
        raise ValueError("Mismatch between enriched data files and textual files")

    sub_file_rich_li = sorted([f"{_}.csv" for _ in sub_file_rich_idx])
    sub_text_file_li = sorted([f"{_}.{extension}" for _ in sub_text_file_idx])

    return zip(sub_file_rich_li, sub_text_file_li)


def generate_window(window_dict, date0_min, date0_max):
    """ generate rolling windows for a set of experiments
    :param window_dict: dictionary of window related parameters
    :param date0_min: earliest date in the enriched data
    :param date0_max: latest date in the enriched data
    """

    trddt = trddt_all[(trddt_all >= date0_min) & (trddt_all <= date0_max)].tolist()
    trdmt = sorted(set([_[:-3] for _ in trddt]))
    trddt_chunck = [[d for d in trddt if d[:-3] == m] for m in trdmt]

    train_win = window_dict["train_win"]
    valid_win = window_dict["valid_win"]
    test_win = window_dict["test_win"]

    def flatten(chunck): return [item for sub_list in chunck for item in sub_list]

    for i in range(len(trddt_chunck) - test_win + 1):
        trddt_train_chunck = trddt_chunck[i: i + train_win]
        trddt_valid_chunck = trddt_chunck[i + train_win: i + valid_win]
        trddt_test_chunck = trddt_chunck[i + valid_win: i + test_win]
        trddt_train = flatten(trddt_train_chunck)
        trddt_valid = flatten(trddt_valid_chunck)
        trddt_test = flatten(trddt_test_chunck)

        yield [trddt_train, trddt_valid, trddt_test]


def generate_params(params_dict, model_name):
    """ generate parameters for an experiment
    :param params_dict: dictionary of models related parameters
    :param model_name: name of the model
    """

    keys, vals_tuple = zip(*params_dict[model_name].items())
    for vals in itertools.product(*vals_tuple):
        params = dict(zip(keys, vals))

        yield params
