from multiprocessing.pool import Pool
from data_prep.data_clean import save_data, clean_data
from data_prep.data_enrich import enrich_data
from data_prep.data_split import split_data
from experiments.backtest import backtest
from experiments.experiment import experiment
from global_settings import CLEAN_PATH, RICH_PATH, DATA_PATH, OUTPUT_PATH, dalym
from main import load_word_sps, load_art_cut, generate_inputs
from experiments.generators import generate_window
from params.params import window_dict, date0_min, date0_max
from params.params import perc_ls
from textual.art_cut import build_art_cut
from textual.word_sps import build_word_sps
from glob import glob
import functools
import os


def run_data_prep(raw_file="raw.csv", data_file="data.csv", clean_file="cleaned.csv"):
    """ Run data processing -- clean, split & enrich data
    :param raw_file: raw file
    :param data_file: data file
    :param clean_file: cleaned file
    """

    # clean & split data
    save_data(raw_file, data_file)
    clean_data(data_file, clean_file)
    split_data(clean_file, split_num=750)

    # define directories
    sub_file_clean_li = [_.split("/")[-1] for _ in glob(os.path.join(CLEAN_PATH, "*.csv"))]
    sub_file_rich_idx = [_.split("/")[-1].split(".")[0].split("_")[1] for _ in glob(os.path.join(RICH_PATH, "*.csv"))]
    sub_file_clean_li = sorted([_ for _ in sub_file_clean_li if _.split(".")[0].split("_")[1] not in sub_file_rich_idx])

    # enrich data
    num_proc = 12
    for idx in range(0, len(sub_file_clean_li), num_proc):
        pool = Pool(num_proc)
        pool.map(enrich_data, sub_file_clean_li[idx: idx + num_proc])
        pool.close()
        pool.join()


def run_textual(textual_name):
    """ Build textual data
    :param textual_name: textual name
    """

    # create textual directory
    text_path = os.path.join(DATA_PATH, textual_name)
    if not os.path.isdir(text_path):
        os.mkdir(text_path)

    # define directories
    extension = "*.npz" if textual_name == "word_sps" else "*.pkl"
    sub_file_rich_li = [_.split("/")[-1] for _ in glob(os.path.join(RICH_PATH, "*.csv"))]
    sub_text_file_idx = [_.split("/")[-1].split(".")[0].split("_")[2] for _ in glob(os.path.join(text_path, extension))]
    sub_file_rich_li = sorted([_ for _ in sub_file_rich_li if _.split(".")[0].split("_")[1] not in sub_text_file_idx])

    # build textual
    num_proc = 12
    build_textual = build_word_sps if textual_name == "word_sps" else build_art_cut
    for idx in range(0, len(sub_file_rich_li), num_proc):
        pool = Pool(num_proc)
        pool.map(build_textual, sub_file_rich_li[idx: idx + num_proc])
        pool.close()
        pool.join()


def run_experiment(model_name):
    """ Run experiment
    :param model_name: model name
    """

    # create model directory
    model_path = os.path.join(OUTPUT_PATH, model_name)
    if not os.path.isdir(model_path):
        os.mkdir(model_path)

    # create sub-directories
    for ev in ["e", "v"]:
        params_sub_path = os.path.join(model_path, f"params_{ev}")
        if not os.path.isdir(params_sub_path):
            os.mkdir(params_sub_path)

        model_sub_path = os.path.join(model_path, f"model_{ev}")
        if not os.path.isdir(model_sub_path):
            os.mkdir(model_sub_path)

    return_sub_path = os.path.join(model_path, "return")
    if not os.path.isdir(return_sub_path):
        os.mkdir(return_sub_path)

    # get input lists
    window_iter = generate_window(window_dict, date0_min, date0_max)
    df_rich, textual = load_word_sps() if model_name == "ssestm" else load_art_cut()
    df_rich_win_iter, textual_win_iter = generate_inputs(window_iter, df_rich, textual)

    # perform experiment
    num_proc = 12
    for idx in range(0, len(window_iter), num_proc):
        pool = Pool(num_proc)
        pool.starmap(
            functools.partial(experiment, model_name=model_name, perc_ls=perc_ls),
            list(zip(
                list(window_iter)[idx: idx + num_proc],
                list(df_rich_win_iter)[idx: idx + num_proc],
                list(textual_win_iter)[idx: idx + num_proc]
            ))
        )
        pool.close()
        pool.join()

    # backtest
    backtest(model_name, dalym)


# if __name__ == "__main__":
#     import os
#     from global_settings import CLEAN_PATH
#     from global_settings import RICH_PATH
#     from global_settings import LOG_PATH
#     PATHS = [CLEAN_PATH, RICH_PATH, LOG_PATH]
#
#     for path in PATHS:
#         if not os.path.isdir(path):
#             os.mkdir(path)


# if __name__ == "__main__":
#     from main import run_data_prep
#     run_data_prep()


# if __name__ == "__main__":
#     from main import run_textual
#     run_textual("word_sps")
#     run_textual("art_cut")


# if __name__ == "__main__":
#     model_name = "ssestm"
#     run_experiment(model_name)
