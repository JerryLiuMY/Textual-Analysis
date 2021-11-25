import numpy as np
import datetime
from experiments.params import params_dict
from experiments.generators import generate_params
from models.ssestm import fit_ssestm, pre_ssestm


def experiment(df_rich, textual, window_iter, model_name):
    """ train models over the sequence of windows and get cumulative return
    :param df_rich: enriched dataframe
    :param textual: textual information (e.g. sparse matrix, embedding)
    :param window_iter: rolling window
    :param model_name: parameters iterator
    :return: equal weighted and value weighted returns
    """

    if model_name == "ssestm":
        fit_func = fit_ssestm
        pre_func = pre_ssestm
    else:
        raise ValueError("Invalid model name")

    ret_e = np.empty(0)
    ret_v = np.empty(0)

    for [trddt_train, trddt_valid, trddt_test] in window_iter:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Training {trddt_train[0][:-3]} to {trddt_train[-1][:-3]}; "
              f"Validation {trddt_valid[0][:-3]} to {trddt_valid[-1][:-3]}; "
              f"Testing {trddt_test[0][:-3]} to {trddt_test[-1][:-3]} ")

        window_idx = df_rich["date_0"].apply(lambda _: _ in trddt_train)
        df_rich_win = df_rich.loc[window_idx, :].reset_index(inplace=False, drop=True)
        textual_win = textual[window_idx, :]
        window = [trddt_train, trddt_valid, trddt_test]
        params_iter = generate_params(params_dict, model_name)
        ret_e_win, ret_v_win = experiment_win(df_rich_win, textual_win, window, params_iter, fit_func, pre_func)
        ret_e = np.concatenate([ret_e, ret_e_win], axis=0)
        ret_v = np.concatenate([ret_v, ret_v_win], axis=0)

    return ret_e, ret_v


def experiment_win(df_rich_win, textual_win, window, params_iter, fit_func, pre_func):
    """ train models over a window and get cumulative return
    :param df_rich_win: enriched dataframe within the window 
    :param textual_win: textual information (e.g. sparse matrix, embedding) within the window
    :param window: [trddt_train, trddt_valid, trddt_test] window
    :param params_iter: parameters iterator
    :param fit_func: parameters iterator
    :param pre_func: parameters iterator
    """

    [trddt_train, trddt_valid, trddt_test] = window
    best_cum_e_scl = -np.inf
    best_cum_v_scl = -np.inf
    best_params_e = dict()
    best_params_v = dict()
    best_model_e = None
    best_model_v = None

    for params in params_iter:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"-Working on {'; '.join([str(key) + '=' + str(value) for key, value in params.items()])}")

        # training
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"--Training...")
        train_idx = df_rich_win["date_0"].apply(lambda _: _ in trddt_train)
        df_rich_win_train = df_rich_win.loc[train_idx, :].reset_index(inplace=False, drop=True)
        textual_win_train = textual_win[train_idx, :]
        model = fit_func(df_rich_win_train, textual_win_train, params)

        # validation
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"--Validation...")
        ret_e_win_valid = np.empty(len(trddt_valid))
        ret_v_win_valid = np.empty(len(trddt_valid))
        for i, dt in enumerate(trddt_valid):
            valid_idx = df_rich_win["date_0"].apply(lambda _: _ == dt)
            df_rich_win_valid = df_rich_win.loc[valid_idx, :].reset_index(inplace=False, drop=True)
            textual_win_valid = textual_win[valid_idx, :]
            ret_e_win_valid[i] = pre_func(df_rich_win_valid, textual_win_valid, params, model, ev="e")
            ret_v_win_valid[i] = pre_func(df_rich_win_valid, textual_win_valid, params, model, ev="v")

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
    print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
          f"-Building return...")
    ret_e_win_test = np.empty(len(trddt_test))
    ret_v_win_test = np.empty(len(trddt_test))
    for i, dt in enumerate(trddt_test):
        test_idx = df_rich_win["date_0"].apply(lambda _: _ == dt)
        df_rich_test = df_rich_win.loc[test_idx, :].reset_index(inplace=False, drop=True)
        textual_test = textual_win[test_idx, :]
        ret_e_win_test[i] = pre_func(df_rich_test, textual_test, best_params_e, best_model_e, ev="e")
        ret_v_win_test[i] = pre_func(df_rich_test, textual_test, best_params_v, best_model_v, ev="v")

    return ret_e_win_test, ret_v_win_test
