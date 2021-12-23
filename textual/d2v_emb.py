import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from scipy.stats import rankdata


def build_d2v_emb(df_rich, doc_cut, params):
    """ compute textual for doc2vec
    :param df_rich: enriched dataframe
    :param doc_cut: articles cut with jieba
    :param params: parameters for doc2vec
    :return: estimated O_hat
    """

    # recover parameters
    window = params["window"]
    vector_size = params["vector_size"]
    epochs = params["epochs"]
    num_bin = params["num_bin"]

    # tag documents
    n = df_rich.shape[0]
    p_hat = (rankdata(df_rich["ret3"].values) - 1) / n
    tag = np.digitize(p_hat, np.linspace(0, 1, num_bin + 1), right=False)
    doc_tag_df = pd.concat([doc_cut, pd.Series(tag, name="tag")], axis=1)
    doc_tag = doc_tag_df.apply(lambda _: TaggedDocument(_["doc_cut"], tags=[_["tag"]]), axis=1)

    # train doc2vec
    doc2vec = Doc2Vec(doc_tag, window=window, vector_size=vector_size, min_count=1, sample=1e-3, workers=4)
    doc2vec.build_vocab(doc_tag)
    doc2vec.train(doc_tag, total_examples=doc2vec.corpus_count, epochs=epochs)
    vec = np.stack(doc_tag.apply(lambda _: doc2vec.infer_vector(_.words, alpha=0.025, epochs=50)).to_numpy())

    return doc2vec, vec
