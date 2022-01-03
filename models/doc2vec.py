from gensim.models.doc2vec import TaggedDocument
from models.classifier import fit_classifier, pre_classifier
from gensim.models import Doc2Vec
from scipy.stats import rankdata
from itertools import tee
import pandas as pd
import numpy as np


def fit_doc2vec(df_rich, art_cut, params):
    """ train classifier
    :param df_rich: enriched dataframe
    :param art_cut: iterable of articles cut with jieba
    :param params: parameters for doc2vec
    :return: the trained doc2vec object and classifier
    """

    # recover parameters
    n = df_rich.shape[0]
    window, vec_size = params["window"], params["vec_size"]
    epochs, num_bins = params["epochs"], params["num_bins"]

    # get inputs
    p_hat = (rankdata(df_rich["ret3"].values) - 1) / n
    target = np.digitize(p_hat, np.linspace(0, 1, num_bins + 1), right=False) - 1
    art_tag_iter = generate_art_tag(art_cut, target)
    art_tag_build, art_tag_train, art_tag_infer = tee(art_tag_iter, 3)

    # train doc2vec
    doc2vec = Doc2Vec(window=window, vector_size=vec_size, epochs=epochs, min_count=5, workers=4)
    doc2vec.build_vocab(art_tag_build)
    doc2vec.train(art_tag_train, total_examples=doc2vec.corpus_count, epochs=doc2vec.epochs)

    # train classifier
    emb_vec = np.empty((0, doc2vec.vector_size), dtype=np.float64)
    for sub_art_tag in art_tag_infer:
        sub_emb_vec = np.vstack(sub_art_tag.apply(lambda _: doc2vec.infer_vector(_.words)).to_numpy())
        emb_vec = np.vstack([emb_vec, sub_emb_vec])
    cls = fit_classifier(emb_vec, target, params)

    return doc2vec, cls


def pre_doc2vec(art_cut, model, params):
    """ predict doc2vec model
    :param art_cut: articles cut with jieba
    :param model: fitted model
    :param params: parameters for doc2vec
    :return: target
    """

    # calculate target
    doc2vec, cls = model
    art_cut = pd.concat(art_cut, axis=0)
    emb_vec = np.vstack(art_cut.apply(lambda _: doc2vec.infer_vector(_)).to_numpy())
    target = pre_classifier(emb_vec, cls, params)

    return target


def generate_art_tag(art_cut, target):
    """ generate article and tag
    :param art_cut: iterable of articles cut with jieba
    :param target: target
    """

    idx = 0
    for sub_art_cut in art_cut:
        sub_target = target[idx: idx + sub_art_cut.shape[0]]
        sub_art_tag_df = pd.concat([sub_art_cut, pd.Series(sub_target, name="tag")], axis=1)
        sub_art_tag = sub_art_tag_df.apply(lambda _: TaggedDocument(words=_["art_cut"], tags=[_["tag"]]), axis=1)
        idx = idx + sub_art_cut.shape[0]

        for line_art_tag in sub_art_tag:
            yield line_art_tag
