from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.losses import CategoricalCrossentropy
from tools.exp_tools import iterable_wrapper
from global_settings import tokenizer
import official.nlp.optimization
from scipy.stats import rankdata
from official import nlp
import tensorflow as tf
import numpy as np


def fit_bert(df_rich, bert_tok, params):
    """ train classifier
    """

    # recover parameters
    n = df_rich.shape[0]
    input_len, batch_size = params["input_len"], params["batch_size"]
    epochs, num_bins = params["epochs"], params["num_bins"]
    steps_per_epoch = int(n / batch_size)
    train_steps = steps_per_epoch * epochs
    warmup_steps = int(train_steps * 0.1)

    # get inputs
    p_hat = (rankdata(df_rich["ret3"].values) - 1) / n
    target = np.digitize(p_hat, np.linspace(0, 1, num_bins + 1), right=False) - 1
    bert_tok_train = generate_art_tag(bert_tok, input_len)

    # retrain model
    model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=num_bins)
    optimizer = nlp.optimization.create_optimizer(2e-5, num_train_steps=train_steps, num_warmup_steps=warmup_steps)
    model.compile(optimizer=optimizer, loss=CategoricalCrossentropy(), metrics=["accuracy"])
    model.fit(bert_tok_train, target=target, batch_size=batch_size, epochs=epochs)

    return model


def pre_bert(doc_cut, model, *args):
    """ predict doc2vec model
    """

    sentiment = None

    return sentiment


@iterable_wrapper
def generate_art_tag(bert_tok, input_len):
    """ generate article and tag
    :param bert_tok: iterable of tokenized text
    :param input_len: length of bert input
    """

    for sub_bert_tok in bert_tok:
        for line_bert_tok in sub_bert_tok:
            for idx in range(0, len(line_bert_tok), input_len - 1):
                input_word_ids = [tokenizer.convert_tokens_to_ids(["[CLS]"])] + line_bert_tok[idx: idx + input_len - 1]
                input_word_ids = tf.ragged.constant(input_word_ids)
                input_mask = tf.ones_like(input_word_ids)
                input_type_ids = tf.zeros_like(input_word_ids)

                input_dict = {
                    "input_word_ids": input_word_ids.to_tensor(),
                    "input_mask": input_mask.to_tensor(),
                    "input_type_ids": input_type_ids.to_tensor()
                }

                yield input_dict
