from transformers import TFAutoModelForSequenceClassification
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from global_settings import DATA_PATH
from tools.exp_tools import iterable_wrapper
from global_settings import tokenizer
from scipy.stats import rankdata
from datetime import datetime
import tensorflow as tf
import numpy as np
import os


def fit_bert(df_rich, bert_tok, params):
    """ train classifier
    """

    # recover parameters
    n = df_rich.shape[0]
    textual_name = "bert_tok"
    textual_path = os.path.join(DATA_PATH, textual_name)
    epochs, num_bins = params["epochs"], params["num_bins"]

    # get inputs
    p_hat = (rankdata(df_rich["ret3"].values) - 1) / n
    target = np.digitize(p_hat, np.linspace(0, 1, num_bins + 1), right=False) - 1
    target = target.reshape(-1, 1)
    batch_train = generate_batch(bert_tok, target, params)

    # retrain model
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Fetching pre-trained BERT model...")
    model = TFAutoModelForSequenceClassification.from_pretrained(
        os.path.join(textual_path, "pre-trained"), num_labels=num_bins
    )
    loss = SparseCategoricalCrossentropy(from_logits=True)
    metrics = [SparseCategoricalAccuracy("accuracy", dtype=tf.float32)]
    model.compile(optimizer=Adam(learning_rate=5e-5), loss=loss, metrics=metrics)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} BERT Training on corpora...")
    model.fit(x=batch_train, epochs=epochs, verbose=0)

    return model


def pre_bert(bert_tok, model, *args):
    """ predict doc2vec model
    """

    target = model.predict(bert_tok)

    return target


@iterable_wrapper
def generate_batch(bert_tok, target, params):
    batch_size = params["batch_size"]
    def init_tensor(input_len): return tf.convert_to_tensor(np.empty((0, input_len), dtype=np.int32))

    def init_batch(input_len):
        init_dict = {
            "input_ids": init_tensor(input_len),
            "attention_mask": init_tensor(input_len),
            "token_type_ids": init_tensor(input_len)
        }
        init_target = np.empty((0, 1))
        
        return init_dict, init_target

    batch_dict, batch_target = init_batch(params["input_len"])
    for idx, (input_dict, input_target) in enumerate(generate_bert_tok(bert_tok, target, params)):
        if idx % batch_size == 0 and idx // batch_size != 0:
            yield batch_dict, batch_target

        if idx % batch_size == 0:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Training on the {idx // batch_size}th batch...")
            batch_dict, batch_target = init_batch(params["input_len"])

        for key in batch_dict.keys():
            batch_dict[key] = tf.concat([batch_dict[key], input_dict[key]], axis=0)
        batch_target = np.concatenate([batch_target, input_target], axis=0)


@iterable_wrapper
def generate_bert_tok(bert_tok, target, params):
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
            input_target = line_target.reshape(-1, 1)
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
