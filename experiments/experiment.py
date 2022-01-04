from params.params import params_dict
from models.ssestm import fit_ssestm, pre_ssestm
from models.doc2vec import fit_doc2vec, pre_doc2vec
from tools.exp_tools import get_return, get_stocks
from tools.exp_tools import get_weights, get_returns
from tools.exp_tools import save_params, save_model, save_return
from models.bert import fit_bert, pre_bert
from experiments.generators import generate_params
from experiments.loader import input_loader
from itertools import tee
from functools import partial
from datetime import datetime
import numpy as np
import pandas as pd
import psutil


def experiment(window, model_name, perc_ls, subset):
    """ Train models over a window and get returns
    :param window: [trddt_train, trddt_valid, trddt_test] window
    :param model_name: model name
    :param perc_ls: percentage of long-short portfolio
    :param subset: whether to use a subset of sample
    """

    # define functions
    [trddt_train, trddt_valid, trddt_test] = window
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
          f"Working on {trddt_train[0][:-3]} to {trddt_test[-1][:-3]} "
          f"({psutil.virtual_memory().percent}% mem used)")

    if model_name == "ssestm":
        load_input = partial(input_loader, textual_name="word_sps", subset=subset)
        fit_func, pre_func = fit_ssestm, pre_ssestm
    elif model_name == "doc2vec":
        load_input = partial(input_loader, textual_name="art_cut", subset=subset)
        fit_func, pre_func = fit_doc2vec, pre_doc2vec
    elif model_name == "bert":
        load_input = partial(input_loader, textual_name="art_cut", subset=subset)
        fit_func, pre_func = fit_bert, pre_bert
    else:
        raise ValueError("Invalid model name")

    # run experiments
    params_iter = generate_params(params_dict, model_name)
    best_cum_e_scl, best_params_e, best_model_e = -np.inf, dict(), tuple()
    best_cum_v_scl, best_params_v, best_model_v = -np.inf, dict(), tuple()

    for params in params_iter:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"* Working on {'; '.join([str(key) + '=' + str(value) for key, value in params.items()])}")

        # training
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"*-* Training {trddt_train[0][:-3]} to {trddt_train[-1][:-3]}...")

        df_rich_win_train, textual_win_train = load_input(trddt_train)
        model = fit_func(df_rich_win_train, textual_win_train, params)

        # validation
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"*-* Validation {trddt_valid[0][:-3]} to {trddt_valid[-1][:-3]}...")

        ret_e_win_valid = np.empty(len(trddt_valid))
        ret_v_win_valid = np.empty(len(trddt_valid))
        for i, dt in enumerate(trddt_valid):
            df_rich_win_valid, textual_win_valid = load_input([dt])
            if df_rich_win_valid.shape[0] == 0:
                ret_e_win_valid[i] = 0.
                ret_v_win_valid[i] = 0.
                continue

            target = pre_func(textual_win_valid, model, params)
            ret_e_win_valid[i] = get_return(df_rich_win_valid, target, perc_ls, "e")[0]
            ret_v_win_valid[i] = get_return(df_rich_win_valid, target, perc_ls, "v")[0]

        cum_e_valid = np.log(np.cumprod(ret_e_win_valid + 1))
        cum_v_valid = np.log(np.cumprod(ret_v_win_valid + 1))

        if cum_e_valid[-1] > best_cum_e_scl:
            best_cum_e_scl = cum_e_valid[-1]
            best_params_e = params
            best_model_e = model

        if cum_v_valid[-1] > best_cum_v_scl:
            best_cum_v_scl = cum_v_valid[-1]
            best_params_v = params
            best_model_v = model

    # out-of-sample testing
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
          f"* Building returns {trddt_test[0][:-3]} to {trddt_test[-1][:-3]}...")

    ret_e_win = np.empty([len(trddt_test), 9], dtype=object)
    ret_v_win = np.empty([len(trddt_test), 9], dtype=object)
    for i, dt in enumerate(trddt_test):
        df_rich_win_test, textual_win_test = load_input([dt])
        if df_rich_win_test.shape[0] == 0:
            ret_e_win[i, 0:6], ret_e_win[i, 6:9] = [np.empty(0)] * 6, [0., 0., 0.]
            ret_v_win[i, 0:6], ret_v_win[i, 6:9] = [np.empty(0)] * 6, [0., 0., 0.]
            continue

        textual_win_test_e, textual_win_test_v = tee(textual_win_test, 2)
        target_e = pre_func(textual_win_test_e, best_model_e, best_params_e)
        target_v = pre_func(textual_win_test_v, best_model_v, best_params_v)
        ret_e_win[i, 0:2] = get_stocks(df_rich_win_test, target_e, perc_ls)
        ret_v_win[i, 0:2] = get_stocks(df_rich_win_test, target_v, perc_ls)
        ret_e_win[i, 2:4] = get_returns(df_rich_win_test, target_e, perc_ls)
        ret_v_win[i, 2:4] = get_returns(df_rich_win_test, target_v, perc_ls)
        ret_e_win[i, 4:6] = get_weights(df_rich_win_test, target_e, perc_ls, "e")
        ret_v_win[i, 4:6] = get_weights(df_rich_win_test, target_v, perc_ls, "v")
        ret_e_win[i, 6:9] = get_return(df_rich_win_test, target_e, perc_ls, "e")
        ret_v_win[i, 6:9] = get_return(df_rich_win_test, target_v, perc_ls, "v")

    # get stocks, weights & returns
    columns_e = ["stks_le", "stks_se", "rets_le", "rets_se", "wgts_le", "wgts_se", "ret_e", "ret_le", "ret_se"]
    columns_v = ["stks_lv", "stks_sv", "rets_lv", "rets_sv", "wgts_lv", "wgts_sv", "ret_v", "ret_lv", "ret_sv"]
    ret_e_win_pkl = pd.DataFrame(ret_e_win[:, 0:6], index=trddt_test, columns=columns_e[0:6])
    ret_v_win_pkl = pd.DataFrame(ret_v_win[:, 0:6], index=trddt_test, columns=columns_v[0:6])
    ret_e_win_csv = pd.DataFrame(ret_e_win[:, 6:9], index=trddt_test, columns=columns_e[6:9])
    ret_v_win_csv = pd.DataFrame(ret_v_win[:, 6:9], index=trddt_test, columns=columns_v[6:9])
    ret_win_pkl = pd.concat([ret_e_win_pkl, ret_v_win_pkl], axis=1)
    ret_win_csv = pd.concat([ret_e_win_csv, ret_v_win_csv], axis=1)

    # save model & parameters
    trddt_test_Ym = datetime.strptime(trddt_test[0], "%Y-%m-%d").strftime("%Y-%m")
    save_model(best_model_e, model_name, trddt_test_Ym, "e")
    save_model(best_model_v, model_name, trddt_test_Ym, "v")
    save_params(best_params_e, model_name, trddt_test_Ym, "e")
    save_params(best_params_v, model_name, trddt_test_Ym, "v")
    save_return(ret_win_pkl, model_name, trddt_test_Ym, ".pkl")
    save_return(ret_win_csv, model_name, trddt_test_Ym, ".csv")
