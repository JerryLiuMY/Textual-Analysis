from sklearn.linear_model import LogisticRegression


def fit_classifier(emb_vec, target, params):
    """ train classifier given emb_vec & sentiment
    :param emb_vec: embedding vector
    :param target: target
    :param params: parameters for the classifier
    """

    # train classifier
    cls = LogisticRegression(n_jobs=4)
    cls.fit(emb_vec, target)

    return cls
