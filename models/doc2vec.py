import numpy as np
from sklearn.linear_model import LogisticRegression


def fit_doc2vec(vec, tag, params):
    """ train classifier
    :param vec: enriched dataframe
    :param tag: articles cut with jieba
    :param params: parameters for the classifier
    :return: the trained classifier
    """

    # recover parameters
    cls = LogisticRegression(n_jobs=4)
    cls.fit(vec, tag)
    model = cls

    return model


def pre_doc2vec(doc_cut, model, params):
    """ predict doc2vec model
    :param doc_cut: cut word
    :param model: fitted model
    :param params: parameters for the classifier
    :return: document tag
    """

    # calculate tag
    doc2vec, logreg = model
    vec = np.stack(doc_cut.apply(lambda _: doc2vec.infer_vector(_, alpha=0.025, epochs=50)).to_numpy())
    tag = logreg.predict(vec)
    sentiment = tag

    return sentiment
