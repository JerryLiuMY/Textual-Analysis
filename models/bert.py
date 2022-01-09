import os
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from global_settings import DATA_PATH
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
from scipy.stats import rankdata
tfds.disable_progress_bar()


def fit_bert(df_rich, bert_tok, params):
    """ train classifier
    """

    # recover parameters
    n = df_rich.shape[0]
    window, vec_size = params["window"], params["vec_size"]
    epochs, num_bins = params["epochs"], params["num_bins"]
    textual_name = "bert_tok"
    textual_path = os.path.join(DATA_PATH, textual_name)

    p_hat = (rankdata(df_rich["ret3"].values) - 1) / n
    target = np.digitize(p_hat, np.linspace(0, 1, num_bins + 1), right=False) - 1

    # build model & restore weights
    bert_config_file = os.path.join(textual_path, "pre-trained", "bert_config.json")
    config_dict = json.loads(tf.io.gfile.GFile(bert_config_file).read())
    bert_config = bert.configs.BertConfig.from_dict(config_dict)
    bert_classifier, bert_encoder = bert.bert_models.classifier_model(bert_config, num_labels=num_bins)
    checkpoint = tf.train.Checkpoint(encoder=bert_encoder)
    checkpoint.read(os.path.join(textual_path, "pre-trained", "bert_model.ckpt")).assert_consumed()

    # setup optimizer
    batch_size = 32
    train_data_size = len(glue_train_labels)
    steps_per_epoch = int(train_data_size / batch_size)
    num_train_steps = steps_per_epoch * epochs
    warmup_steps = int(num_train_steps * 0.1)
    optimizer = nlp.optimization.create_optimizer(2e-5, num_train_steps=num_train_steps, num_warmup_steps=warmup_steps)

    # train the model
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy("accuracy", dtype=tf.float32)]
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    bert_classifier.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    bert_classifier.fit(
        glue_train, glue_train_labels,
        validation_data=(glue_validation, glue_validation_labels),
        batch_size=batch_size,
        epochs=epochs
    )

    return model


def pre_bert(doc_cut, model, *args):
    """ predict doc2vec model
    """

    sentiment = None

    return sentiment
