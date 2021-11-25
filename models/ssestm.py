import numpy as np
from sklearn.preprocessing import Normalizer
import pandas as pd
from global_settings import full_dict


def fit_ssestm(df_rich, word_sps, *args):
    """ train ssestm model to get the estimated O_hat
    :param df_rich: enriched dataframe
    :param word_sps: sparse word count matrix
    :return: estimated O_hat
    """

    # Get D_hat and W_hat
    n = df_rich.shape[0]
    normalizer = Normalizer(norm="l1")
    D_hat = normalizer.fit_transform(word_sps).T.toarray()
    p_hat = np.argsort(df_rich["ret3"].values).reshape(1, -1) / n
    W_hat = np.concatenate((p_hat, 1 - p_hat))

    # Calculate O_hat
    O_hat = D_hat @ W_hat.T @ np.linalg.inv(W_hat @ W_hat.T)
    O_hat = O_hat.clip(min=0)
    O_hat = np.divide(O_hat, O_hat.sum(axis=0))

    return O_hat


def predict_ssestm(df_rich, word_sps, param, O_hat):
    """ predict p_hat based on the word_matrix and the estimated O_hat
    :param df_rich: enriched dataframe
    :param word_sps: word_matrix
    :param param: parameters for ssestm
    :param O_hat: estimated O_hat
    :return: p_hat values for the samples in the word_matrix
    """

    pen = param["pen"]

    # Get D_hat and W_lin
    normalizer = Normalizer(norm="l1")
    D_hat = normalizer.fit_transform(word_sps).T.toarray()
    p_lin = np.linspace(0, 1, 1000)[1:-1]
    W_lin = np.array([p_lin, 1 - p_lin])

    # Calculate p_hat
    likelihood = D_hat.T @ np.log(O_hat @ W_lin)
    penalty = pen * np.log(W_lin[0, :] * W_lin[1, :]).reshape(1, -1)
    objective = likelihood + penalty
    p_hat = p_lin[np.argmax(objective, axis=1)]

    num_stocks = int(len(p_hat) * 0.05)
    ret_le = df_rich.iloc[np.argsort(p_hat)[-num_stocks:], :]["ret"].mean()
    ret_se = df_rich.iloc[np.argsort(p_hat)[:num_stocks], :]["ret"].mean()
    ret_lv = df_rich.iloc[np.argsort(p_hat)[-num_stocks:], :]["ret"].mean()
    ret_sv = df_rich.iloc[np.argsort(p_hat)[:num_stocks], :]["ret"].mean()

    return ret_le, ret_se, ret_lv, ret_sv
