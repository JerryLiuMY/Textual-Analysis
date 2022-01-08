import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from official.modeling import tf_utils
from official import nlp
from official.nlp import bert
import official.nlp.optimization
import official.nlp.bert.bert_models
import official.nlp.bert.configs
import official.nlp.bert.run_classifier
import official.nlp.bert.tokenization
import official.nlp.data.classifier_data_lib
import official.nlp.modeling.losses
import official.nlp.modeling.models
import official.nlp.modeling.networks
tfds.disable_progress_bar()


def fit_bert(vec, tag, params):
    """ train classifier
    """

    model = None

    return model


def pre_bert(doc_cut, model, *args):
    """ predict doc2vec model
    """

    sentiment = None

    return sentiment
