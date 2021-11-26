import json
import os
import numpy as np
from global_settings import OUTPUT_PATH


def save_params(params, model_name, trddt_test, ev):
    """ Save model parameters
    :param params: parameters to be saved
    :param model_name: model name
    :param trddt_test: testing trading dates
    :param ev: equal/value weighted type
    """
    params_path = os.path.join(OUTPUT_PATH, model_name)
    params_sub_path = os.path.join(params_path, f"params_{ev}")

    with open(os.path.join(params_sub_path, f"{trddt_test[0][:-3]}.json"), "w") as f:
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
        np.save(os.path.join(model_sub_path, f"{trddt_test[0][:-3]}.npy"), model)


def get_window(window_iter, trddt_test_Ym):
    """ Getting window from trddt_test in the format of %Y-%m
    :param window_iter: window iterator
    :param trddt_test_Ym: trddt_test in the format of %Y-%m
    """

    for [trddt_train, trddt_valid, trddt_test] in window_iter:
        if trddt_test[0][:-3] == trddt_test_Ym:
            window = [trddt_train, trddt_valid, trddt_test]
            return window
