import numpy as np


def train_ssestm(df_rich, word_df):
    article_filter = (word_df.sum(axis=1) != 0)
    word_df = word_df.loc[article_filter, :]
    df_rich = df_rich.loc[article_filter, :]
    n = word_df.shape[0]

    D_hat = word_df.div(word_df.sum(axis=1), axis=0).values.T
    p_hat = np.argsort(df_rich["ret3"].values).reshape(1, -1) / n
    W_hat = np.concatenate((p_hat, 1 - p_hat))
    O_hat = D_hat @ W_hat.T @ np.linalg.inv(W_hat @ W_hat.T)
    O_hat = O_hat.clip(min=0)
    O_hat = np.divide(O_hat, O_hat.sum(axis=0))

    return O_hat


def predict_ssestm(O_hat, word_df, pen):
    np.linspace()
    p_li = np.linspace(0, 1, 1000)[1:-1]
    D = word_df.div(word_df.sum(axis=1), axis=0).values.T
    S = np.diag(1 / word_df.sum(axis=1).values)
    likelihood = S @ D.T @ np.log(O_hat @ np.array([p_li, 1 - p_li])) + pen*np.log(p_li * (1 - p_li)).reshape(1, -1)

