from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
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
    epochs, num_bins = params["epochs"], params["num_bins"]

    # get inputs
    p_hat = (rankdata(df_rich["ret3"].values) - 1) / n
    target = np.digitize(p_hat, np.linspace(0, 1, num_bins + 1), right=False) - 1
    target = target.reshape(-1, 1)
    bert_tok_train = generate_art_tag(bert_tok, target, params)

    # retrain model
    model = TFAutoModelForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=num_bins)
    optimizer = nlp.optimization.create_optimizer(init_lr=2e-5)
    loss = SparseCategoricalCrossentropy(from_logits=True)
    metrics = [SparseCategoricalAccuracy("accuracy", dtype=tf.float32)]
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    model.fit(x=bert_tok_train, epochs=epochs, verbose=1)

    return model


def pre_bert(bert_tok, model, *args):
    """ predict doc2vec model
    """

    target = model.predict(bert_tok)

    return target


@iterable_wrapper
def generate_art_tag(bert_tok, target, params):
    """ generate article and tag
    :param bert_tok: iterable of tokenized text
    :param target: sentiment target
    :param params: parameters
    """

    idx = 0
    input_len = params["input_len"]

    for sub_bert_tok in bert_tok:
        sub_target = target[idx: idx + sub_bert_tok.shape[0], :]
        for line_bert_tok, line_target in zip(sub_bert_tok, sub_target):
            input_target = line_target
            for foo in range(0, len(line_bert_tok), input_len - 1):
                input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"]) + line_bert_tok[foo: foo + input_len - 1]
                input_ids = tf.expand_dims(tf.convert_to_tensor(input_ids), axis=0)
                attention_mask = tf.ones_like(input_ids)
                token_type_ids = tf.zeros_like(input_ids)

                current_len = input_ids.shape[1]
                input_ids = tf.pad(input_ids, [[0, 0], [0, input_len - current_len]], "CONSTANT")
                attention_mask = tf.pad(attention_mask, [[0, 0], [0, input_len - current_len]], "CONSTANT")
                token_type_ids = tf.pad(token_type_ids, [[0, 0], [0, input_len - current_len]], "CONSTANT")

                input_dict = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids
                }

                yield input_dict, input_target
