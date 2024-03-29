from sklearn.linear_model import LogisticRegression
from tensorflow.keras.models import Sequential
from global_settings import OUTPUT_PATH
from scipy.stats.stats import pearsonr
import pickle
import numpy as np
import pandas as pd
import json
import os


def iterable_wrapper(func):
    """ Wrap generator to be reusable """
    def inner(*args, **kwargs):
        return IterableWrapper(func, *args, **kwargs)
    return inner


class IterableWrapper:
    """ Wrap generator to be reusable """
    def __init__(self, func, *args, **kwargs):
        self.func, self.args, self.kwargs = func, args, kwargs
        self.generator = self.func(*self.args, **self.kwargs)

    def __iter__(self):
        self.generator = self.func(*self.args, **self.kwargs)
        return self

    def __len__(self):
        length = 0
        for _ in self.generator:
            length += 1
        return length

    def __next__(self):
        item = next(self.generator)
        if item is not None:
            return item
        else:
            raise StopIteration


def save_model(model, model_name, trddt_test_Ym, ev):
    """ Save trained model
    :param model: model to be saved
    :param model_name: model name
    :param trddt_test_Ym: testing trading dates Ym
    :param ev: equal/value weighted type
    """

    model_path = os.path.join(OUTPUT_PATH, model_name)
    model_sub_path = os.path.join(model_path, f"model_{ev}")

    if model_name == "simple":
        pass
    elif model_name == "ssestm":
        np.save(os.path.join(model_sub_path, f"{trddt_test_Ym}.npy"), model)
    elif model_name == "doc2vec":
        doc2vec, cls = model
        doc2vec.save(os.path.join(model_sub_path, f"{trddt_test_Ym}.model"))
        if isinstance(cls, LogisticRegression):
            with open(os.path.join(model_sub_path, f"{trddt_test_Ym}.pkl"), "wb") as f:
                pickle.dump(cls, f)
        elif isinstance(cls, Sequential):
            cls.save(os.path.join(model_sub_path, f"{trddt_test_Ym}.h5"))
        else:
            raise ValueError("Invalid classifier type")
    elif model_name == "bert":
        model.save_pretrained(os.path.join(model_sub_path, f"{trddt_test_Ym}"))
    else:
        raise ValueError("Invalid model name")


def save_params(params, model_name, trddt_test_Ym, ev):
    """ Save model parameters
    :param params: parameters to be saved
    :param model_name: model name
    :param trddt_test_Ym: testing trading dates Ym
    :param ev: equal/value weighted type
    """

    model_path = os.path.join(OUTPUT_PATH, model_name)
    params_sub_path = os.path.join(model_path, f"params_{ev}")

    with open(os.path.join(params_sub_path, f"{trddt_test_Ym}.json"), "w") as f:
        json.dump(params, f)


def save_return(return_df, model_name, trddt_test_Ym, extension):
    """ Save return dataframe
    :param return_df: return dataframe
    :param model_name: model name
    :param trddt_test_Ym: testing trading dates Ym
    :param extension: file extension
    """

    model_path = os.path.join(OUTPUT_PATH, model_name)
    return_sub_path = os.path.join(model_path, "return")

    if extension == ".pkl":
        return_df.to_pickle(os.path.join(return_sub_path, f"{trddt_test_Ym}.pkl"), compression=None, protocol=4)
    elif extension == ".csv":
        return_df.to_csv(os.path.join(return_sub_path, f"{trddt_test_Ym}.csv"))
    else:
        raise ValueError("Invalid extension")


def get_rich_ls(df_rich, target, perc_ls):
    """ Get long/short dataframes
    :param df_rich: enriched dataframe
    :param target: predicted target for portfolio construction
    :param perc_ls: percentage of L/S portfolio
    :return df_rich_l: long enriched dataframe
    :return df_rich_s: short enriched dataframe
    """

    # group by stock_mention
    df_rich_cp = df_rich.copy(deep=True)
    df_rich_cp.loc[:, "target"] = target
    df_rich_gb = df_rich_cp.groupby("stock_mention")

    # return if no stock to L/S
    num_ls = int(len(df_rich_gb) * perc_ls)
    if num_ls == 0:
        df_rich_l = pd.DataFrame(columns=df_rich_cp.columns)
        df_rich_s = pd.DataFrame(columns=df_rich_cp.columns)
        return df_rich_l, df_rich_s

    # get L/S keys
    keys = list(df_rich_gb.groups.keys())
    sorted_idx = np.argsort(df_rich_gb["target"].mean().to_numpy())
    keys_l = [keys[i] for i in sorted_idx[-num_ls:]]
    keys_s = [keys[i] for i in sorted_idx[:num_ls]]

    # get L/S dataframes (first occurrence in each group)
    df_rich_l = df_rich_cp.loc[df_rich_cp.apply(lambda _: _["stock_mention"] in keys_l, axis=1), :]
    df_rich_s = df_rich_cp.loc[df_rich_cp.apply(lambda _: _["stock_mention"] in keys_s, axis=1), :]
    df_rich_l = df_rich_l.groupby("stock_mention").first().reset_index(inplace=False, drop=False)
    df_rich_s = df_rich_s.groupby("stock_mention").first().reset_index(inplace=False, drop=False)
    df_rich_l = df_rich_l.loc[:, df_rich_cp.columns]
    df_rich_s = df_rich_s.loc[:, df_rich_cp.columns]

    return df_rich_l, df_rich_s


def get_stocks(df_rich, target, perc_ls):
    """ Get L/S stocks from the predicted targets
    :param df_rich: enriched dataframe
    :param target: predicted target for portfolio construction
    :param perc_ls: percentage of L/S portfolio
    :return stks_l, stks_s: L/S stocks
    """

    # df_rich.shape[0] == 0 automatically accounted for
    df_rich_l, df_rich_s = get_rich_ls(df_rich, target, perc_ls)
    stks_l = df_rich_l.loc[:, "stock_mention"].to_numpy()
    stks_s = df_rich_s.loc[:, "stock_mention"].to_numpy()

    return stks_l, stks_s


def get_returns(df_rich, target, perc_ls):
    """ Get L/S returns from the predicted targets
    :param df_rich: enriched dataframe
    :param target: predicted target for portfolio construction
    :param perc_ls: percentage of L/S portfolio
    :return rets_l, rets_s: L/S returns
    """

    # df_rich.shape[0] == 0 automatically accounted for
    df_rich_l, df_rich_s = get_rich_ls(df_rich, target, perc_ls)
    rets_l = df_rich_l.loc[:, "ret"].to_numpy()
    rets_s = df_rich_s.loc[:, "ret"].to_numpy()

    return rets_l, rets_s


def get_weights(df_rich, target, perc_ls, ev):
    """ Get L/S weights from the predicted targets
    :param df_rich: enriched dataframe
    :param target: predicted target for portfolio construction
    :param perc_ls: percentage of L/S portfolio
    :param ev: equal vs. value weighted type
    :return wgts_l, wgts_s: L/S weights
    """

    df_rich_l, df_rich_s = get_rich_ls(df_rich, target, perc_ls)
    if df_rich_l.shape[0] == df_rich_s.shape[0] == 0:
        return np.empty(0), np.empty(0)

    if ev == "e":
        wgts_l = (1 / df_rich_l.shape[0]) * np.ones(df_rich_l.shape[0])
        wgts_s = (1 / df_rich_s.shape[0]) * np.ones(df_rich_s.shape[0])
    elif ev == "v":
        wgts_l = (df_rich_l["cap"] / np.sum(df_rich_l["cap"])).to_numpy()
        wgts_s = (df_rich_s["cap"] / np.sum(df_rich_s["cap"])).to_numpy()
    else:
        raise ValueError("Invalid weighting type")

    return wgts_l, wgts_s


def get_return(df_rich, target, perc_ls, ev):
    """ Get return and L/S return from the predicted targets
    :param df_rich: enriched dataframe
    :param target: predicted target for portfolio construction
    :param perc_ls: percentage of L/S portfolio
    :param ev: equal vs. value weighted type
    :return ret, ret_l, ret_s: total return and L/S return
    """

    df_rich_l, df_rich_s = get_rich_ls(df_rich, target, perc_ls)
    if df_rich_l.shape[0] == df_rich_s.shape[0] == 0:
        return 0., 0., 0.

    if ev == "e":
        ret_l = +df_rich_l["ret"].mean()
        ret_s = -df_rich_s["ret"].mean()
    elif ev == "v":
        ret_l = +np.average(df_rich_l["ret"], weights=df_rich_l["cap"])
        ret_s = -np.average(df_rich_s["ret"], weights=df_rich_s["cap"])
    else:
        raise ValueError("Invalid weighting type")

    ret = ret_l + ret_s

    return ret, ret_l, ret_s


def get_pearsonr(df_rich, target):
    """ Get Pearson correlation coefficient
    :param df_rich: enriched dataframe
    :param target: predicted target for portfolio construction
    :return pearson_cor: Pearson correlation coefficient
    """

    df_rich_cp = df_rich.copy(deep=True)
    df_rich_cp.loc[:, "target"] = target
    df_rich_gb = df_rich_cp.groupby("stock_mention")
    cor = pearsonr(df_rich_gb["target"].mean().to_numpy(), df_rich_gb["ret"].first().to_numpy())[0]

    return cor
