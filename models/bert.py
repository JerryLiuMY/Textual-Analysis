import os
import torch
import numpy as np
from tqdm.auto import tqdm
from datetime import datetime
from torch.nn import functional
from global_settings import DATA_PATH
from global_settings import tokenizer
from tools.exp_tools import iterable_wrapper
from scipy.stats import rankdata
from transformers import AdamW
from transformers import get_scheduler
from transformers import AutoModelForSequenceClassification


def fit_bert(df_rich, bert_tok, params):
    """ train classifier
    :param df_rich: enriched dataframe
    :param bert_tok: iterable of bert tokens
    :param params: parameters for bert
    :return: the trained bert classifier
    """

    # recover parameters
    n = df_rich.shape[0]
    textual_name = "bert_tok"
    textual_path = os.path.join(DATA_PATH, textual_name)
    trained_path = os.path.join(textual_path, "pre-trained")
    num_bins, epochs = params["num_bins"], params["epochs"]

    # get inputs
    p_hat = (rankdata(df_rich["ret3"].values) - 1) / n
    target = np.digitize(p_hat, np.linspace(0, 1, num_bins + 1), right=False) - 1
    target = target.reshape(-1, 1)
    batch_train = generate_batch(bert_tok, target, params)
    steps_per_epoch = len(batch_train)

    # retrain model
    model = AutoModelForSequenceClassification.from_pretrained(trained_path, num_labels=num_bins)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = epochs * steps_per_epoch
    lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} BERT Training on corpora...")
    model.to(device)
    model.train()
    for epoch in range(epochs):
        progress_bar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}")
        for batch in batch_train:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    return model


def pre_bert(bert_tok, model, *args):
    """ predict doc2vec model
    """

    target = model.predict(bert_tok)

    return target


@iterable_wrapper
def generate_batch(bert_tok, target, params):
    """ generate bert tokens by batch
    :param bert_tok: iterable of tokenized text
    :param target: sentiment target
    :param params: parameters
    """

    batch_size = params["batch_size"]
    def init_tensor(input_len): return torch.tensor(np.empty((0, input_len), dtype=np.int32))

    def init_batch(input_len):
        init_dict = {
            "input_ids": init_tensor(input_len),
            "attention_mask": init_tensor(input_len),
            "token_type_ids": init_tensor(input_len),
            "labels": np.empty((0, 1))
        }

        return init_dict

    batch_dict = init_batch(params["input_len"])
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Generating the 0th batch...")
    for idx, input_dict in enumerate(generate_bert_tok(bert_tok, target, params)):
        if idx % batch_size == 0 and idx // batch_size != 0:
            yield batch_dict

        if idx % batch_size == 0 and idx // batch_size != 0:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Generating the {idx // batch_size}th batch...")
            batch_dict = init_batch(params["input_len"])

        for key in batch_dict.keys():
            if isinstance(batch_dict[key], torch.Tensor):
                batch_dict[key] = torch.cat((batch_dict[key], input_dict[key]), dim=0)
            elif isinstance(batch_dict[key], np.ndarray):
                batch_dict[key] = np.concatenate([batch_dict[key], input_dict[key]], axis=0)
            else:
                raise ValueError("Invalid data type")


@iterable_wrapper
def generate_bert_tok(bert_tok, target, params):
    """ generate bert tokens by line
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
                current_len = len(input_ids)

                input_ids = torch.tensor(input_ids).expand(1, current_len)
                attention_mask = torch.ones_like(input_ids)
                token_type_ids = torch.zeros_like(input_ids)

                input_ids = functional.pad(input_ids, (0, input_len - current_len), "constant", 0)
                attention_mask = functional.pad(attention_mask, (0, input_len - current_len), "constant", 0)
                token_type_ids = functional.pad(token_type_ids, (0, input_len - current_len), "constant", 0)

                input_dict = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                    "labels": input_target,
                }

                yield input_dict
