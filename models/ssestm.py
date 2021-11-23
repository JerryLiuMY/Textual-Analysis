import numpy as np
import pandas as pd
from tools.params import full_dict
from tools.params import params_dict


def train_ssestm(df_rich, word_matrix):
    """ train ssestm model to get the O_hat estimate
    :param df_rich: enriched dataframe
    :param word_matrix: word count matrix
    :return: estimated O_hat
    """

    word_df = pd.DataFrame(word_matrix, columns=full_dict)
    article_filter = (word_df.sum(axis=1) != 0)
    word_df = word_df.loc[article_filter, :]
    df_rich = df_rich.loc[article_filter, :]

    # Get D_hat and W_hat
    n = word_df.shape[0]
    D_hat = word_df.div(word_df.sum(axis=1), axis=0).values.T
    p_hat = np.argsort(df_rich["ret3"].values).reshape(1, -1) / n
    W_hat = np.concatenate((p_hat, 1 - p_hat))

    # Calculate O hat
    O_hat = D_hat @ W_hat.T @ np.linalg.inv(W_hat @ W_hat.T)
    O_hat = O_hat.clip(min=0)
    O_hat = np.divide(O_hat, O_hat.sum(axis=0))

    return O_hat


def predict_ssestm(word_matrix, O_hat, params):
    """
    :param word_matrix:
    :param O_hat: estimated O_hat
    :param params: parameters for ssestm
    :return:
    """

    pen = params["pen"]
    word_df = pd.DataFrame(word_matrix, columns=full_dict)
    D = word_df.div(word_df.sum(axis=1), axis=0).values.T
    p = np.linspace(0, 1, 1000)[1:-1]
    penalty = pen * np.log(p * (1 - p)).reshape(1, -1)
    likelihood = D.T @ np.log(O_hat @ np.array([p, 1 - p])) + penalty
    p_hat = p[np.argmax(likelihood, axis=1)]

    return p_hat
