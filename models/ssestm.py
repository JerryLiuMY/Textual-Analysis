import numpy as np
import pandas as pd
from global_settings import full_dict


def train_ssestm(df_rich, word_mtx):
    """ train ssestm model to get the estimated O_hat
    :param df_rich: enriched dataframe
    :param word_mtx: word count matrix
    :return: estimated O_hat
    """

    word_df = pd.DataFrame(word_mtx, columns=full_dict)
    article_filter = (word_df.sum(axis=1) != 0)
    word_df = word_df.loc[article_filter, :]
    df_rich = df_rich.loc[article_filter, :]

    # Get D_hat and W_hat
    n = word_df.shape[0]
    D_hat = word_df.div(word_df.sum(axis=1), axis=0).values.T
    p_hat = np.argsort(df_rich["ret3"].values).reshape(1, -1) / n
    W_hat = np.concatenate((p_hat, 1 - p_hat))

    # Calculate O_hat
    O_hat = D_hat @ W_hat.T @ np.linalg.inv(W_hat @ W_hat.T)
    O_hat = O_hat.clip(min=0)
    O_hat = np.divide(O_hat, O_hat.sum(axis=0))

    return O_hat


def predict_ssestm(df_rich, word_matrix, O_hat, params):
    """ predict p_hat based on the word_matrix and the estimated O_hat
    :param df_rich: enriched dataframe
    :param word_matrix: word_matrix
    :param O_hat: estimated O_hat
    :param params: parameters for ssestm
    :return: p_hat values for the samples in the word_matrix
    """

    pen = params["pen"]
    word_df = pd.DataFrame(word_matrix, columns=full_dict)

    # Get D_hat and W_lin
    D_hat = word_df.div(word_df.sum(axis=1), axis=0).values.T
    p_lin = np.linspace(0, 1, 1000)[1:-1]
    W_lin = np.array([p_lin, 1 - p_lin])

    # Calculate p_hat
    likelihood = D_hat.T @ np.log(O_hat @ W_lin)
    penalty = pen * np.log(W_lin[0, :] * W_lin[1, :]).reshape(1, -1)
    objective = likelihood + penalty
    p_hat = p_lin[np.argmax(objective, axis=1)]

    return p_hat
