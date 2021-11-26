import os
import numpy as np
import pandas as pd
from experiments.params import params_dict
from experiments.generators import generate_params
from models.ssestm import fit_ssestm, pre_ssestm
from global_settings import OUTPUT_PATH
from tools.exp_tools import save_params, save_model, get_return
from datetime import datetime


def experiment(df_rich, textual, window_iter, model_name, perc_ls):
    """ train models over a sequence of windows and get cumulative return
    :param df_rich: enriched dataframe
    :param textual: textual information (e.g. sparse matrix, embedding)
    :param window_iter: rolling window
    :param model_name: parameters iterator
    :param perc_ls: percentage of long-short portfolio
    :return ret_e_win: equal weighted return (ret, ret_l, ret_s) with shape=[len(trddt), 3]
    :return ret_v_win: value weighted returns (ret, ret_l, ret_s) with shape=[len(trddt), 3]
    """

    # create directories
    model_path = os.path.join(OUTPUT_PATH, model_name)
    for ev in ["e", "v"]:
        params_sub_path = os.path.join(model_path, f"params_{ev}")
        if not os.path.isdir(params_sub_path):
            os.mkdir(params_sub_path)

        model_sub_path = os.path.join(model_path, f"model_{ev}")
        if not os.path.isdir(model_sub_path):
            os.mkdir(model_sub_path)

    # define functions
    if model_name == "ssestm":
        fit_func = fit_ssestm
        pre_func = pre_ssestm
    else:
        raise ValueError("Invalid model name")

    # define dataframes
    ret_e_df = pd.DataFrame(columns=["ret_e", "ret_le", "ret_se"])
    ret_v_df = pd.DataFrame(columns=["ret_v", "ret_lv", "ret_sv"])

    for [trddt_train, trddt_valid, trddt_test] in window_iter:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Working on {trddt_train[0][:-3]} to {trddt_test[-1][:-3]}")

        trddt_window = trddt_train + trddt_valid + trddt_test
        window_idx = df_rich["date_0"].apply(lambda _: _ in trddt_window)
        df_rich_win = df_rich.loc[window_idx, :].reset_index(inplace=False, drop=True)
        textual_win = textual[window_idx, :]
        window = [trddt_train, trddt_valid, trddt_test]
        params_iter = generate_params(params_dict, model_name)
        outputs = experiment_win(df_rich_win, textual_win, window, params_iter, fit_func, pre_func, perc_ls)
        ret_e_win, ret_v_win = outputs[0]
        best_params_e, best_params_v = outputs[1]
        best_model_e, best_model_v = outputs[2]

        # save returns
        ret_e_win_df = pd.DataFrame(ret_e_win, index=trddt_test, columns=["ret_e", "ret_le", "ret_se"])
        ret_v_win_df = pd.DataFrame(ret_v_win, index=trddt_test, columns=["ret_v", "ret_lv", "ret_sv"])
        ret_e_df = pd.concat([ret_e_df, ret_e_win_df], axis=0)
        ret_v_df = pd.concat([ret_v_df, ret_v_win_df], axis=0)

        # save parameters
        save_params(best_params_e, model_name, trddt_test, "e")
        save_params(best_params_v, model_name, trddt_test, "v")
        save_model(best_model_e, model_name, trddt_test, "e")
        save_model(best_model_v, model_name, trddt_test, "v")

    ret_df = pd.concat([ret_e_df, ret_v_df], axis=1)
    # noinspection PyTypeChecker
    ret_df.to_csv(os.path.join(model_path, "ret_df.csv"))


def experiment_win(df_rich_win, textual_win, window, params_iter, fit_func, pre_func, perc_ls):
    """ train models over a window and get cumulative return
    :param df_rich_win: enriched dataframe within the window 
    :param textual_win: textual information (e.g. sparse matrix, embedding) within the window
    :param window: [trddt_train, trddt_valid, trddt_test] window
    :param params_iter: parameters iterator
    :param fit_func: parameters iterator
    :param pre_func: parameters iterator
    :param perc_ls: percentage of long-short portfolio
    :return ret_e_win: equal weighted return (ret, ret_l, ret_s) with shape=[len(trddt_win_test), 3]
    :return ret_v_win: value weighted returns (ret, ret_l, ret_s) with shape=[len(trddt_win_test), 3]
    """

    [trddt_train, trddt_valid, trddt_test] = window
    best_cum_e_scl = -np.inf
    best_cum_v_scl = -np.inf
    best_params_e = dict()
    best_params_v = dict()
    best_model_e = None
    best_model_v = None

    for params in params_iter:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"* Working on {'; '.join([str(key) + '=' + str(value) for key, value in params.items()])}")

        # training
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"*-* Training {trddt_train[0][:-3]} to {trddt_train[-1][:-3]}...")
        train_idx = df_rich_win["date_0"].apply(lambda _: _ in trddt_train)
        df_rich_win_train = df_rich_win.loc[train_idx, :].reset_index(inplace=False, drop=True)
        textual_win_train = textual_win[train_idx, :]
        model = fit_func(df_rich_win_train, textual_win_train, params)

        # validation
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"*-* Validation {trddt_valid[0][:-3]} to {trddt_valid[-1][:-3]}...")
        ret_e_win_valid = np.empty(len(trddt_valid))
        ret_v_win_valid = np.empty(len(trddt_valid))
        for i, dt in enumerate(trddt_valid):
            valid_idx = df_rich_win["date_0"].apply(lambda _: _ == dt)
            if sum(valid_idx) == 0:
                ret_e_win_valid[i] = 0
                ret_v_win_valid[i] = 0
                continue

            df_rich_win_valid = df_rich_win.loc[valid_idx, :].reset_index(inplace=False, drop=True)
            textual_win_valid = textual_win[valid_idx, :]
            target = pre_func(textual_win_valid, params, model)
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

    # building returns
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
          f"* Building returns {trddt_test[0][:-3]} to {trddt_test[-1][:-3]}...")
    ret_e_win = np.empty([len(trddt_test), 3])
    ret_v_win = np.empty([len(trddt_test), 3])
    for i, dt in enumerate(trddt_test):
        test_idx = df_rich_win["date_0"].apply(lambda _: _ == dt)
        if sum(test_idx) == 0:
            ret_e_win[i] = 0
            ret_v_win[i] = 0
            continue

        df_rich_win_test = df_rich_win.loc[test_idx, :].reset_index(inplace=False, drop=True)
        textual_win_test = textual_win[test_idx, :]
        target_e = pre_func(textual_win_test, best_params_e, best_model_e)
        target_v = pre_func(textual_win_test, best_params_v, best_model_v)
        ret_e_win[i, :] = get_return(df_rich_win_test, target_e, perc_ls, "e")
        ret_v_win[i, :] = get_return(df_rich_win_test, target_v, perc_ls, "v")

    return (ret_e_win, ret_v_win), (best_params_e, best_params_v), (best_model_e, best_model_v)
