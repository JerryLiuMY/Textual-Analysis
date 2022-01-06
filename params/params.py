perc_ls = 0.05
subset_size = 0.1

window_dict = {
    "train_win": 12,
    "valid_win": 12 + 6,
    "test_win": 12 + 6 + 1
}

proc_dict = {
    "ssestm": 37,
    "doc2vec": 1,
    "bert": 5,
}

params_dict = {
    "ssestm": {"pen": [0.0]},
    "doc2vec": {"window": [10], "vec_size": [20], "epochs": [20], "num_bins": [20], "cls_type": ["lr"]},
    "bert": {},
    "dnn": {"hidden": [60]},
}
