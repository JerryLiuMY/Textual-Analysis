import numpy as np


def ssestm(df_rich, word_df):
    article_filter = (word_df.sum(axis=1) != 0)
    word_df = word_df.loc[article_filter, :]
    df_rich = word_df.loc[article_filter, :]
    n = word_df.shape[0]

    p_hat = np.argsort(df_rich["ret3"].values).reshape(1, -1) / n
    W = np.concatenate((p_hat, 1 - p_hat))
    O = D @ W.T @ np.linalg.inv(W @ W.T)

    pass