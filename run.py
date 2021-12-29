from multiprocessing.pool import Pool
from multiprocessing import Process
from data_prep.data_clean import save_data, clean_data
from data_prep.data_enrich import enrich_data
from data_prep.data_split import split_data
from experiments.backtest import backtest
from experiments.experiment import experiment
from main import load_word_sps, load_art_cut, build_inputs
from global_settings import DATA_PATH, OUTPUT_PATH, dalym
from global_settings import CLEAN_PATH, RICH_PATH, LOG_PATH
from experiments.generators import generate_window
from params.params import window_dict, date0_min, date0_max
from params.params import perc_ls
from textual.art_cut import build_art_cut
from textual.word_sps import build_word_sps
from glob import glob
import os


def run_data_prep(raw_file="raw.csv", data_file="data.csv", clean_file="cleaned.csv"):
    """ Run data processing -- clean, split & enrich data
    :param raw_file: raw file
    :param data_file: data file
    :param clean_file: cleaned file
    """

    # create directories
    for path in [CLEAN_PATH, RICH_PATH, LOG_PATH]:
        if not os.path.isdir(path):
            os.mkdir(path)

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

    # create directory
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


def run_experiment(model_name, idx_from, idx_to, if_subset):
    """ Run experiment
    :param model_name: model name
    :param idx_from: start of the index of the full window list
    :param idx_to: end of the index of the full window list
    :param if_subset: whether to use subset of data
    """

    # create directory
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

    # perform experiment
    window_full = list(generate_window(window_dict, date0_min, date0_max))
    df_rich, textual = load_word_sps(if_subset) if model_name == "ssestm" else load_art_cut(if_subset)

    procs = []
    for window in window_full[idx_from: idx_to]:
        df_rich_win, textual_win = build_inputs(window, df_rich, textual)
        proc = Process(target=experiment, args=(window, df_rich_win, textual_win, model_name, perc_ls))
        procs.append(proc)

    for proc in procs:
        proc.start()

    for proc in procs:
        proc.join()


def run_backtest(model_name):
    """ Run backtest
    :param model_name: model name
    """

    # backtest
    backtest(model_name, dalym)


if __name__ == "__main__":
    # run_data_prep()
    # run_textual("word_sps")
    # run_textual("art_cut")
    pass


if __name__ == "__main__":
    import argparse
    model_name = "doc2vec"
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("-f", "--idx_from", type=int, help="Testing window initial year")
    parser.add_argument("-t", "--idx_to", type=int, help="Testing window final year")
    args = parser.parse_args()
    run_experiment(model_name, idx_from=args.idx_from, idx_to=args.idx_to, if_subset=True)

if __name__ == "__main__":
    model_name = "doc2vec"
    backtest(model_name, dalym)
