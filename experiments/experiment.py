import numpy as np
import pandas as pd
from datetime import datetime
from params.params import params_dict
from models.ssestm import fit_ssestm, pre_ssestm
from models.doc2vec import fit_doc2vec, pre_doc2vec
from models.bert import fit_bert, pre_bert
from tools.exp_tools import get_textual, get_df_rich, get_return
from tools.exp_tools import get_stocks, get_weights, get_returns
from tools.exp_tools import save_params, save_model, save_return
from experiments.generators import generate_params
import psutil


def experiment(window, df_rich_win, textual_win, model_name, perc_ls):
    """ train models over a window and get cumulative return
    :param window: [trddt_train, trddt_valid, trddt_test] window
    :param df_rich_win: enriched dataframe within the window 
    :param textual_win: textual information (e.g. sparse matrix, embeddings) within the window
    :param model_name: model name
    :param perc_ls: percentage of long-short portfolio
    :return ret_e_win: equal weighted returns (ret, ret_l, ret_s) with shape=[len(trddt_win_test), 3]
    :return ret_v_win: value weighted returns (ret, ret_l, ret_s) with shape=[len(trddt_win_test), 3]
    """

    # define functions
    [trddt_train, trddt_valid, trddt_test] = window
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
          f"Working on {trddt_train[0][:-3]} to {trddt_test[-1][:-3]} "
          f"({psutil.virtual_memory().percent}% mem used)")

    if model_name == "ssestm":
        fit_func, pre_func = fit_ssestm, pre_ssestm
    elif model_name == "doc2vec":
        fit_func, pre_func = fit_doc2vec, pre_doc2vec
    elif model_name == "bert":
        fit_func, pre_func = fit_bert, pre_bert
    else:
        raise ValueError("Invalid model name")

    # run experiments
    params_iter = generate_params(params_dict, model_name)
    best_cum_e_scl, best_params_e, best_model_e = -np.inf, dict(), None
    best_cum_v_scl, best_params_v, best_model_v = -np.inf, dict(), None

    for params in params_iter:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"* Working on {'; '.join([str(key) + '=' + str(value) for key, value in params.items()])}")

        # training
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"*-* Training {trddt_train[0][:-3]} to {trddt_train[-1][:-3]}...")

        train_idx = df_rich_win["date_0"].apply(lambda _: _ in trddt_train)
        df_rich_win_train = get_df_rich(df_rich_win, train_idx)
        textual_win_train = get_textual(textual_win, train_idx)
        model = fit_func(df_rich_win_train, textual_win_train, params)

        # validation
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"*-* Validation {trddt_valid[0][:-3]} to {trddt_valid[-1][:-3]}...")

        ret_e_win_valid = np.empty(len(trddt_valid))
        ret_v_win_valid = np.empty(len(trddt_valid))
        for i, dt in enumerate(trddt_valid):
            valid_idx = df_rich_win["date_0"].apply(lambda _: _ == dt)
            if sum(valid_idx) == 0:
                ret_e_win_valid[i], ret_v_win_valid[i] = 0., 0.
                continue

            df_rich_win_valid = get_df_rich(df_rich_win, valid_idx)
            textual_win_valid = get_textual(textual_win, valid_idx)
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
        test_idx = df_rich_win["date_0"].apply(lambda _: _ == dt)
        if sum(test_idx) == 0:
            ret_e_win[i, 0:6], ret_e_win[i, 6:9] = [np.empty(0)] * 6, [0., 0., 0.]
            ret_v_win[i, 0:6], ret_v_win[i, 6:9] = [np.empty(0)] * 6, [0., 0., 0.]
            continue

        df_rich_win_test = get_df_rich(df_rich_win, test_idx)
        textual_win_test = get_textual(textual_win, test_idx)
        target_e = pre_func(textual_win_test, best_model_e, best_params_e)
        target_v = pre_func(textual_win_test, best_model_v, best_params_v)
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
