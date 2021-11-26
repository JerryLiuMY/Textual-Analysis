import json
import os
import numpy as np
from global_settings import OUTPUT_PATH
from datetime import datetime


def get_return(df_rich, target, perc_ls, ev):
    """ Get returns from the predicted p-hat values
    :param df_rich: enriched dataframe
    :param target: predicted target for portfolio construction
    :param perc_ls: equal vs. value weighted type
    :param ev: equal vs. value weighted type
    """

    # Calculate equal and value weighted returns
    num_ls = int(len(target) * perc_ls)
    sorted_idx = np.argsort(target)
    df_rich_l = df_rich.iloc[sorted_idx[-num_ls:], :]
    df_rich_s = df_rich.iloc[sorted_idx[:num_ls], :]

    if ev == "e":
        ret_l = df_rich_l["ret"].mean()
        ret_s = -df_rich_s["ret"].mean()
    elif ev == "v":
        ret_l = np.average(df_rich_l["ret"], weights=df_rich_l["cap"])
        ret_s = -np.average(df_rich_s["ret"], weights=df_rich_s["cap"])
    else:
        raise ValueError("Invalid weighting type")

    ret = ret_l + ret_s

    return ret, ret_l, ret_s


def save_params(params, model_name, trddt_test, ev):
    """ Save model parameters
    :param params: parameters to be saved
    :param model_name: model name
    :param trddt_test: testing trading dates
    :param ev: equal/value weighted type
    """

    params_path = os.path.join(OUTPUT_PATH, model_name)
    params_sub_path = os.path.join(params_path, f"params_{ev}")

    trddt_test_Ym = datetime.strptime(trddt_test[0], "%Y-%m-%d").strftime("%Y-%m")
    with open(os.path.join(params_sub_path, f"{trddt_test_Ym}.json"), "w") as f:
        json.dump(params, f)


def save_model(model, model_name, trddt_test, ev):
    """ Save trained model
    :param model: model to be saved
    :param model_name: model name
    :param trddt_test: testing trading dates
    :param ev: equal/value weighted type
    """

    model_path = os.path.join(OUTPUT_PATH, model_name)
    model_sub_path = os.path.join(model_path, f"model_{ev}")

    if model_name == "ssestm":
        trddt_test_Ym = datetime.strptime(trddt_test[0], "%Y-%m-%d").strftime("%Y-%m")
        np.save(os.path.join(model_sub_path, f"{trddt_test_Ym}.npy"), model)


def get_window(window_iter, trddt_test_Ym):
    """ Getting window from trddt_test in the format of %Y-%m
    :param window_iter: window iterator
    :param trddt_test_Ym: trddt_test in the format of %Y-%m
    """

    for [trddt_train, trddt_valid, trddt_test] in window_iter:
        if datetime.strptime(trddt_test[0], "%Y-%m-%d").strftime("%Y-%m") == trddt_test_Ym:
            window = [trddt_train, trddt_valid, trddt_test]
            return window
