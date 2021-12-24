import numpy as np
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from models.classifier import fit_classifier
from scipy.stats import rankdata
import pandas as pd


def fit_doc2vec(df_rich, art_cut, params):
    """ train classifier
    :param df_rich: enriched dataframe
    :param art_cut: articles cut with jieba
    :param params: parameters for doc2vec
    :return: the trained classifier
    """

    # recover parameters
    window, vec_size = params["window"], params["vec_size"]
    epochs, num_bins = params["epochs"], params["num_bins"]
    n = df_rich.shape[0]

    # tag article & define target
    art_tag_df = pd.concat([art_cut, pd.Series(df_rich.index, name="tag")], axis=1)
    art_tag = art_tag_df.apply(lambda _: TaggedDocument(_["art_tag"], tags=[_["tag"]]), axis=1)
    p_hat = (rankdata(df_rich["ret3"].values) - 1) / n
    target = np.digitize(p_hat, np.linspace(0, 1, num_bins + 1), right=False)

    # train doc2vec
    doc2vec = Doc2Vec(art_tag, dm=0, window=window, vector_size=vec_size, min_count=1, sample=1e-3, workers=4)
    doc2vec.build_vocab(art_tag)
    doc2vec.train(art_tag, total_examples=doc2vec.corpus_count, epochs=epochs)
    emb_vec = np.stack(art_tag.apply(lambda _: doc2vec.infer_vector(_.words, alpha=0.025, epochs=50)).to_numpy())
    cls = fit_classifier(emb_vec, target, params)

    return doc2vec, cls


def pre_doc2vec(doc_cut, model, params):
    """ predict doc2vec model
    :param doc_cut: cut word
    :param model: fitted model
    :param params: parameters for the classifier
    :return: document tag
    """

    # calculate tag
    doc2vec, logreg = model
    emb_vec = np.stack(doc_cut.apply(lambda _: doc2vec.infer_vector(_, alpha=0.025, epochs=50)).to_numpy())
    target = logreg.predict(emb_vec)

    return target
