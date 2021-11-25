import numpy as np
from global_settings import perc_ls
from sklearn.preprocessing import Normalizer


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
    model = O_hat

    return model


def pre_ssestm(df_rich, word_sps, params, model, ev):
    """ predict p_hat based on the word_matrix and the estimated O_hat
    :param df_rich: enriched dataframe
    :param word_sps: word_matrix
    :param params: parameters for ssestm
    :param model: fitted model
    :param ev: equal vs. value weighted type
    :return: p_hat values for the samples in the word_matrix
    """

    # recover parameters
    pen = params["pen"]
    O_hat = model

    # Get D_hat and W_lin
    normalizer = Normalizer(norm="l1")
    D_hat = normalizer.fit_transform(word_sps).T.toarray()
    p_lin = np.linspace(0, 1, 1000)[1:-1]
    W_lin = np.array([p_lin, 1 - p_lin])

    # Calculate p_hat
    likelihood = D_hat.T @ np.log(O_hat @ W_lin)
    penalty = pen * np.log(W_lin[0, :] * W_lin[1, :]).reshape(1, -1)
    objective = likelihood + penalty
    p_hat = np.take(p_lin, np.argmax(objective, axis=1))

    # Calculate equal and value weighted returns
    num_ls = int(len(p_hat) * 0.05)
    sorted_idx = np.argsort(p_hat)
    df_rich_l = df_rich.iloc[sorted_idx[-num_ls:], :]
    df_rich_s = df_rich.iloc[sorted_idx[:num_ls], :]

    if ev == "e":
        ret_l = df_rich_l["ret"].mean()
        ret_s = df_rich_s["ret"].mean()
    elif ev == "v":
        ret_l = np.average(df_rich_l["ret"], weights=df_rich_l["cap"])
        ret_s = np.average(df_rich_s["ret"], weights=df_rich_s["cap"])
    else:
        raise ValueError("Invalid weighting type")

    ret = ret_l - ret_s

    return ret
