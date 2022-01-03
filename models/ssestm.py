from sklearn.preprocessing import Normalizer
from scipy.sparse import csr_matrix
from scipy.stats import rankdata
import numpy as np
import scipy as sp


def fit_ssestm(df_rich, word_sps, *args):
    """ train ssestm model to get the estimated O_hat
    :param df_rich: enriched dataframe
    :param word_sps: sparse word count matrix
    :return: estimated O_hat
    """

    # get D_hat and W_hat
    n, normalizer = df_rich.shape[0], Normalizer(norm="l1")
    word_sps = sp.sparse.vstack(word_sps, format="csr")
    D_hat = normalizer.fit_transform(word_sps).T
    p_hat = (rankdata(df_rich["ret3"].values) - 1).reshape(1, -1) / n

    # calculate O_hat
    W_hat = np.concatenate((p_hat, 1 - p_hat))
    O_hat = D_hat @ csr_matrix(W_hat.T @ np.linalg.inv(W_hat @ W_hat.T))
    O_hat = O_hat.toarray()
    O_hat = O_hat.clip(min=0)
    O_hat = np.divide(O_hat, O_hat.sum(axis=0))

    return O_hat


def pre_ssestm(word_sps, model, params):
    """ predict p_hat based on the word_matrix and the estimated O_hat
    :param word_sps: word_matrix
    :param model: fitted model
    :param params: parameters for ssestm
    :return: p_hat values for the samples in the word_matrix
    """

    # recover parameters
    pen = params["pen"]
    O_hat = model

    # get D_hat and W_lin
    word_sps = sp.sparse.vstack(word_sps, format="csr")
    zero_idx = (np.sum(O_hat, axis=1) == 0.0)
    normalizer = Normalizer(norm="l1")
    O_hat = O_hat[~zero_idx, :]
    D_hat = normalizer.fit_transform(word_sps).T
    D_hat = D_hat[~zero_idx, :]
    p_lin = np.linspace(0, 1, 1000)[1:-1]
    W_lin = np.array([p_lin, 1 - p_lin])

    # calculate p_hat
    likelihood = D_hat.T @ csr_matrix(np.log(O_hat @ W_lin))
    penalty = pen * np.log(W_lin[0, :] * W_lin[1, :]).reshape(1, -1)
    objective = likelihood.toarray() + penalty
    p_hat = np.take(p_lin, np.argmax(objective, axis=1))

    return p_hat
