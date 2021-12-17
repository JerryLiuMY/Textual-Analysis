from sklearn.linear_model import LogisticRegression


def fit_cls(vec, tag, params):
    """ train classifier
    :param vec: enriched dataframe
    :param tag: articles cut with jieba
    :param params: parameters for the classifier
    :return: the trained classifier
    """

    # recover parameters
    cls = LogisticRegression(n_jobs=4)
    cls.fit(vec, tag)

    return cls
