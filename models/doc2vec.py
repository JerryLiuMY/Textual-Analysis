from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from models.classifier import fit_classifier
from scipy.stats import rankdata
import pandas as pd
import numpy as np


def fit_doc2vec(df_rich, art_cut, params):
    """ train classifier
    :param df_rich: enriched dataframe
    :param art_cut: articles cut with jieba
    :param params: parameters for doc2vec
    :return: the trained classifier
    """

    # recover parameters
    n = df_rich.shape[0]
    window, vec_size = params["window"], params["vec_size"]
    epochs, num_bins = params["epochs"], params["num_bins"]

    # article & target
    art_tag_df = pd.concat([art_cut, pd.Series(df_rich.index, name="tag")], axis=1)
    art_tag = art_tag_df.apply(lambda _: TaggedDocument(_["art_cut"], tags=[_["tag"]]), axis=1)
    p_hat = (rankdata(df_rich["ret3"].values) - 1) / n
    target = np.digitize(p_hat, np.linspace(0, 1, num_bins + 1), right=False) - 1

    # train doc2vec
    doc2vec = Doc2Vec(window=window, vector_size=vec_size, epochs=epochs, min_count=5, workers=4)
    doc2vec.build_vocab(art_tag)
    doc2vec.train(art_tag, total_examples=doc2vec.corpus_count, epochs=doc2vec.epochs)

    # train classifier
    emb_vec = np.vstack(art_tag.apply(lambda _: doc2vec.infer_vector(_.words)).to_numpy())
    cls = fit_classifier(emb_vec, target, params)

    return doc2vec, cls


def pre_doc2vec(art_cut, model, params):
    """ predict doc2vec model
    :param art_cut: articles cut with jieba
    :param model: fitted model
    :param params: parameters for doc2vec
    :return: document tag
    """

    # calculate target
    doc2vec, cls = model
    emb_vec = np.vstack(art_cut.apply(lambda _: doc2vec.infer_vector(_)).to_numpy())

    cls_type = params["cls_type"]

    if cls_type == "lr":
        target = cls.predict(emb_vec)
    elif cls_type == "mlp":
        target = np.argmax(cls.predict(emb_vec), axis=1)
    else:
        raise ValueError("Invalid classifier type")

    return target
