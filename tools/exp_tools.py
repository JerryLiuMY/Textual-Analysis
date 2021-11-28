import json
import os
import numpy as np
from global_settings import OUTPUT_PATH
from datetime import datetime
import joblib
import pandas as pd


def get_textual(textual, idx):
    """ Get textual data from boolean array
    :param textual: textual data
    :param idx: boolean array of index
    """

    if isinstance(textual, np.ndarray):
        return textual[idx]
    elif isinstance(textual, pd.Series):
        return textual.loc[idx].reset_index(inplace=False, drop=True)
    elif isinstance(textual, pd.DataFrame):
        return textual.loc[idx, :].reset_index(inplace=False, drop=True)
    else:
        raise ValueError("Textual size not recognized")


def save_model(model, model_name, trddt_test, ev):
    """ Save trained model
    :param model: model to be saved
    :param model_name: model name
    :param trddt_test: testing trading dates
    :param ev: equal/value weighted type
    """

    model_path = os.path.join(OUTPUT_PATH, model_name)
    model_sub_path = os.path.join(model_path, f"model_{ev}")
    trddt_test_Ym = datetime.strptime(trddt_test[0], "%Y-%m-%d").strftime("%Y-%m")

    if model_name == "ssestm":
        np.save(os.path.join(model_sub_path, f"{trddt_test_Ym}_O_hat.npy"), model)
    elif model_name == "doc2vec":
        doc2vec, logreg = model[0], model[1]
        doc2vec.save(os.path.join(model_sub_path, f"{trddt_test_Ym}_doc2vec.npy"))
        joblib.dump(logreg, os.path.join(model_sub_path, f"{trddt_test_Ym}_logreg.joblib"))
    else:
        raise ValueError("Invalid model name")


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


def get_return(df_rich, target, perc_ls, ev):
    """ Get returns from the predicted p-hat values
    :param df_rich: enriched dataframe
    :param target: predicted target for portfolio construction
    :param perc_ls: equal vs. value weighted type
    :param ev: equal vs. value weighted type
    """

    # Calculate equal and value weighted returns
    num_ls = int(len(target) * perc_ls)
    if num_ls == 0:
        return 0., 0., 0.

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


# def get_window(window_iter, trddt_test_Ym):
#     """ Get window from trddt_test in the format of %Y-%m
#     :param window_iter: window iterator
#     :param trddt_test_Ym: trddt_test in the format of %Y-%m
#     """
#
#     for [trddt_train, trddt_valid, trddt_test] in window_iter:
#         if datetime.strptime(trddt_test[0], "%Y-%m-%d").strftime("%Y-%m") == trddt_test_Ym:
#             window = [trddt_train, trddt_valid, trddt_test]
#             return window
