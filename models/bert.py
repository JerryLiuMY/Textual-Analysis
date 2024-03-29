import os
import torch
import numpy as np
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
    """ fine-tune BERT classifier
    :param df_rich: enriched dataframe
    :param bert_tok: iterable of bert tokens
    :param params: parameters for bert
    :return: trained bert classifier
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
    batch_build = generate_batch(bert_tok, target, params)
    batch_train = generate_batch(bert_tok, target, params)

    # setting up configs
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} BERT Preparing inputs...")
    steps_per_epoch = len(batch_build)
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} BERT Setting up configs...")
    model = AutoModelForSequenceClassification.from_pretrained(trained_path, num_labels=num_bins)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = epochs * steps_per_epoch
    lr_scheduler = get_scheduler("linear", optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # training BERT model
    GPUs = [torch.cuda.get_device_name(_) for _ in range(torch.cuda.device_count())]
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} GPUs: {GPUs}...")
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} BERT Training on corpora...")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.train()
    for epoch in range(epochs):
        for batch in batch_train:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

    return model


def pre_bert(bert_tok, model, params):
    """ predict bert model
    :param bert_tok: iterable of bert tokens
    :param model: fitted model
    :param params: parameters for bert
    :return: target
    """

    input_len = params["input_len"]

    target = []
    # recover parameters
    for sub_bert_tok in bert_tok:
        for line_bert_tok in sub_bert_tok:
            t_li = []
            for foo in range(0, len(line_bert_tok), input_len - 1):
                input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"]) + line_bert_tok[foo: foo + input_len - 1]
                current_len = len(input_ids)

                input_ids = torch.tensor(input_ids).expand(1, current_len)
                attention_mask = torch.ones_like(input_ids)
                token_type_ids = torch.zeros_like(input_ids)

                input_ids = functional.pad(input_ids, (0, input_len - current_len), "constant", 0)
                attention_mask = functional.pad(attention_mask, (0, input_len - current_len), "constant", 0)
                token_type_ids = functional.pad(token_type_ids, (0, input_len - current_len), "constant", 0)

                # define input dictionary
                input_dict = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "token_type_ids": token_type_ids,
                }

                t_ = model.predict(input_dict)
                t_li.append(t_)

            target.append(np.mean(t_li))

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
            "labels": torch.tensor(np.empty(0, dtype=np.int32))
        }

        return init_dict

    batch_dict = None
    for idx, input_dict in enumerate(generate_bert_tok(bert_tok, target, params)):
        if idx % batch_size == 0 and idx // batch_size != 0:
            yield batch_dict

        if idx % batch_size == 0:
            print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} Generating the {idx // batch_size}th batch...")
            batch_dict = init_batch(params["input_len"])

        for key in batch_dict.keys():
            batch_dict[key] = torch.cat((batch_dict[key], input_dict[key]), dim=0)


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
        sub_target = target[idx: idx + sub_bert_tok.shape[0]]
        for line_bert_tok, line_target in zip(sub_bert_tok, sub_target):
            input_target = torch.tensor([line_target])
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
