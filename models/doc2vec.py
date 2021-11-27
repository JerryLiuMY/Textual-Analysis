import pandas as pd
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument


def fit_doc2vec(df_rich, doc_cut, params):
    """ train doc2vec model
    :param df_rich: enriched dataframe
    :param doc_cut: articles cut with jieba
    :param params: parameters for doc2vec
    :return: estimated O_hat
    """

    # Get D_hat and W_hat
    num_bin = 20
    n = df_rich.shape[0]
    p_hat = np.argsort(df_rich["ret3"].values) / n
    tag = np.digitize(p_hat, np.linspace(0, 1, num_bin + 1), right=False)

    # tag documents
    def tag_doc(df): return TaggedDocument(df["doc_cut"], tags=[df["tag"]])
    doc_tag_df = pd.concat([doc_cut, pd.Series(tag, name="tag")], axis=1)
    doc_tag = doc_tag_df.apply(tag_doc, axis=1)

    # train model
    window = params["window"]
    vector_size = params["vector_size"]
    model = Doc2Vec(doc_tag, min_count=1, window=window, vector_size=vector_size, sample=1e-3, nagative=5, workers=4)
    model.train(doc_tag, total_examples=model.corpus_count, epochs=10)

    return model


def pre_doc2vec(doc_cut, model, *args):
    """ predict doc2vec model
    :param doc_cut: cut word
    :param model: fitted model
    :return: document tag
    """

    vector = model.infer_vector(doc_words=doc_cut, alpha=0.025, steps=500)

