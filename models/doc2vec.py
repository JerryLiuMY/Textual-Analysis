from models.classifier import fit_classifier, pre_classifier
from gensim.models.doc2vec import TaggedDocument
from tools.exp_tools import iterable_wrapper
from gensim.models import Doc2Vec
from scipy.stats import rankdata
from datetime import datetime
import pandas as pd
import numpy as np
import logging


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
    tag = list(df_rich.index)
    p_hat = (rankdata(df_rich["ret3"].values) - 1) / n
    target = np.digitize(p_hat, np.linspace(0, 1, num_bins + 1), right=False) - 1
    art_tag_build = generate_art_tag(art_cut, tag)
    art_tag_train = generate_art_tag(art_cut, tag)

    # train doc2vec
    logging.basicConfig(format="%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S", level=logging.INFO)
    doc2vec = Doc2Vec(window=window, vector_size=vec_size, epochs=epochs, min_count=10, sample=1e-05, workers=4)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Gensim Doc2Vec Building vocabulary...")
    doc2vec.build_vocab(art_tag_build)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Gensim Doc2Vec Training on corpora...")
    doc2vec.train(art_tag_train, total_examples=doc2vec.corpus_count, epochs=doc2vec.epochs)

    # train classifier
    logging.getLogger().setLevel(logging.DEBUG)
    emb_vec = np.vstack([doc2vec.dv[[_]] for _ in tag])
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
    art_cut = pd.concat(art_cut, axis=0).reset_index(inplace=False, drop=True)
    emb_vec = np.vstack(art_cut.apply(lambda _: doc2vec.infer_vector(_)).to_numpy())
    target = pre_classifier(emb_vec, cls, params)

    return target


@iterable_wrapper
def generate_art_tag(art_cut, tag):
    """ generate article and tag
    :param art_cut: iterable of articles cut with jieba
    :param tag: article tag
    """

    idx = 0
    for sub_art_cut in art_cut:
        sub_tag = tag[idx: idx + sub_art_cut.shape[0]]
        sub_art_tag_df = pd.concat([sub_art_cut, pd.Series(sub_tag, name="tag")], axis=1)
        sub_art_tag = sub_art_tag_df.apply(lambda _: TaggedDocument(words=_["art_cut"], tags=[_["tag"]]), axis=1)
        idx = idx + sub_art_cut.shape[0]

        for line_art_tag in sub_art_tag:
            yield line_art_tag
