from data_prep.data_clean import save_data, clean_data
from data_prep.data_enrich import enrich_data
from data_prep.data_split import split_data
from experiments.backtest import backtest
from experiments.experiment import experiment
from global_settings import DATA_PATH, OUTPUT_PATH, dalym
from global_settings import CLEAN_PATH, RICH_PATH, LOG_PATH
from global_settings import date0_min, date0_max
from experiments.generator import generate_window
from params.params import window_dict
from params.params import perc_ls
from params.params import proc_dict
from textuals.word_sps import build_word_sps
from textuals.art_cut import build_art_cut
from textuals.bert_tok import build_bert_tok
from multiprocessing.pool import Pool
from multiprocessing import Process
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
    split_data(clean_file)

    # define directories
    sub_file_clean_li = [_.split("/")[-1] for _ in glob(os.path.join(CLEAN_PATH, "*.csv"))]
    sub_file_rich_idx = [_.split("/")[-1].split(".")[0] for _ in glob(os.path.join(RICH_PATH, "*.csv"))]
    sub_file_clean_li = sorted([_ for _ in sub_file_clean_li if _.split(".")[0] not in sub_file_rich_idx])

    # enrich data
    num_proc = 18
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
    textual_path = os.path.join(DATA_PATH, textual_name)
    if not os.path.isdir(textual_path):
        os.mkdir(textual_path)

    # define directories
    extension = "npz" if textual_name == "word_sps" else "pkl"
    sub_file_rich_li = [_.split("/")[-1] for _ in glob(os.path.join(RICH_PATH, "*.csv"))]
    sub_text_file_idx = [_.split("/")[-1].split(".")[0] for _ in glob(os.path.join(textual_path, "*." + extension))]
    sub_file_rich_li = sorted([_ for _ in sub_file_rich_li if _.split(".")[0] not in sub_text_file_idx])

    # build textual
    if textual_name == "word_sps":
        build_textual = build_word_sps
    elif textual_name == "art_cut":
        build_textual = build_art_cut
    elif textual_name == "bert_tok":
        build_textual = build_bert_tok
    else:
        raise ValueError("Invalid textual name")

    num_proc = 18
    for idx in range(0, len(sub_file_rich_li), num_proc):
        pool = Pool(num_proc)
        pool.map(build_textual, sub_file_rich_li[idx: idx + num_proc])
        pool.close()
        pool.join()


def run_experiment(model_name, subset):
    """ Run experiment
    :param model_name: model name
    :param subset: whether to use a subset of data
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
    window_li = list(generate_window(window_dict, date0_min, date0_max))
    num_proc = proc_dict[model_name]

    for idx in range(0, len(window_li), num_proc):
        windows, procs = window_li[idx: idx + num_proc], []
        for window in windows:
            proc = Process(target=experiment, args=(window, model_name, perc_ls, subset))
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
    # run_textual("bert_tok")
    # run_experiment("simple", subset=False)
    run_backtest("doc2vec")


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Run experiment")
#     parser.add_argument("-m", "--model_name", type=str, help="Model name")
#     parser.add_argument("-s", "--subset", nargs="?", default=False, const=True, help="Use subset of data")
#     args = parser.parse_args()
#
#     run_experiment(model_name=args.model_name, subset=args.subset)
#     run_backtest(model_name=args.model_name)
