from global_settings import full_dict
from scipy.sparse import csr_matrix
from scipy.stats import rankdata
import numpy as np
import scipy as sp


def fit_simple(df_rich, word_sps, *args):
    """ train a simple model to get the counts
    :param df_rich: enriched dataframe
    :param word_sps: iterable of sparse word count matrix
    :return: computed model
    """

    return None


def pre_simple(word_sps, model, params):
    """ predict p_hat based on the word_matrix
    :param word_sps: iterable of sparse word count matrix
    :param model: fitted model
    :param params: parameters for simple model
    :return: p_hat values for the samples in the word_matrix
    """

    index_z = full_dict.index("涨")
    index_d = full_dict.index("跌")
    word_sps = sp.sparse.vstack(word_sps, format="csr")
    diff_mtx = csr_matrix.todense(word_sps[:, index_z] - word_sps[:, index_d])
    diff_arr = np.array(diff_mtx).reshape(-1)
    p_hat = (rankdata(diff_arr) - 1) / len(diff_arr)

    return p_hat
