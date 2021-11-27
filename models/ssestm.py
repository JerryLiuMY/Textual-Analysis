import numpy as np
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
    p_hat = np.argsort(df_rich["ret3"].values).reshape(1, -1) / n - 0.5

    # zero_idx = (np.sum(D_hat, axis=0) == 0)
    # D_hat = D_hat[:, ~zero_idx]
    # p_hat = p_hat[:, ~zero_idx]

    # Calculate O_hat
    W_hat = np.concatenate((p_hat, 1 - p_hat))
    O_hat = D_hat @ W_hat.T @ np.linalg.inv(W_hat @ W_hat.T)
    O_hat = O_hat.clip(min=0)
    O_hat = np.divide(O_hat, O_hat.sum(axis=0))
    model = O_hat

    return model


def pre_ssestm(word_sps, params, model):
    """ predict p_hat based on the word_matrix and the estimated O_hat
    :param word_sps: word_matrix
    :param params: parameters for ssestm
    :param model: fitted model
    :return: p_hat values for the samples in the word_matrix
    """

    # Recover parameters
    pen = params["pen"]
    O_hat = model

    # Get D_hat and W_lin
    zero_idx = (np.sum(O_hat, axis=1) == 0.0)
    normalizer = Normalizer(norm="l1")
    O_hat = O_hat[~zero_idx, :]
    D_hat = normalizer.fit_transform(word_sps).T
    D_hat = D_hat[~zero_idx, :].toarray()
    p_lin = np.linspace(0, 1, 1000)[1:-1]
    W_lin = np.array([p_lin, 1 - p_lin])

    # Calculate p_hat
    likelihood = D_hat.T @ np.log(O_hat @ W_lin)
    penalty = pen * np.log(W_lin[0, :] * W_lin[1, :]).reshape(1, -1)
    objective = likelihood + penalty
    p_hat = np.take(p_lin, np.argmax(objective, axis=1))

    return p_hat
