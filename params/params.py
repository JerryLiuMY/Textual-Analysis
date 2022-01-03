perc_ls = 0.05

window_dict = {
    "train_win": 12,
    "valid_win": 12 + 6,
    "test_win": 12 + 6 + 1
}

proc_dict = {
    "ssestm": 10,
    "doc2vec": 5,
    "bert": 8,
}

params_dict = {
    "ssestm": {"pen": [0.0]},
    "doc2vec": {"window": [10], "vec_size": [15], "epochs": [10], "num_bins": [20], "cls_type": ["lr"], "hidden": [60]},
    "bert": {},
}
