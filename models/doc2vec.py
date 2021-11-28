import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.linear_model import LogisticRegression


def fit_doc2vec(df_rich, doc_cut, params):
    """ train doc2vec
    :param df_rich: enriched dataframe
    :param doc_cut: articles cut with jieba
    :param params: parameters for doc2vec
    :return: estimated O_hat
    """

    # get document tags
    n = df_rich.shape[0]
    num_bin = 20
    p_hat = np.argsort(df_rich["ret3"].values) / n
    tag = np.digitize(p_hat, np.linspace(0, 1, num_bin + 1), right=False)

    # tag documents
    doc_tag_df = pd.concat([doc_cut, pd.Series(tag, name="tag")], axis=1)
    doc_tag = doc_tag_df.apply(lambda _: TaggedDocument(_["doc_cut"], tags=[_["tag"]]), axis=1)

    # train doc2vec
    window = params["window"]
    vector_size = params["vector_size"]
    epochs = params["epochs"]
    doc2vec = Doc2Vec(doc_tag, window=window, vector_size=vector_size, min_count=1, sample=1e-3, workers=4)
    doc2vec.build_vocab(doc_tag)
    doc2vec.train(doc_tag, total_examples=doc2vec.corpus_count, epochs=epochs)
    vec = np.stack(doc_tag.apply(lambda _: doc2vec.infer_vector(_.words, alpha=0.025, epochs=50)).to_numpy())

    # train logistic regression
    tag = doc_tag.apply(lambda _: _.tags[0]).to_numpy()
    logreg = LogisticRegression(n_jobs=4)
    logreg.fit(vec, tag)

    return doc2vec, logreg


def pre_doc2vec(doc_cut, model, *args):
    """ predict doc2vec model
    :param doc_cut: cut word
    :param model: fitted model
    :return: document tag
    """

    # calculate tag
    doc2vec, logreg = model
    vec = np.stack(doc_cut.apply(lambda _: doc2vec.infer_vector(_, alpha=0.025, epochs=50)).to_numpy())
    tag = logreg.predict(vec)

    return tag
