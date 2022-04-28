perc_ls = 0.05
subset_size = 0.1

window_dict = {
    "train_win": 12,
    "valid_win": 12 + 6,
    "test_win": 12 + 6 + 1
}

proc_dict = {
    "ssestm": 37,
    "doc2vec": 13,
    "bert": 1,
}

params_dict = {
    "ssestm": {"pen": [0.0]},
    "doc2vec": {"window": [5, 10, 20], "vec_size": [5, 10, 20], "epochs": [1],
                "cls_type": ["lr", "mlp"], "num_bins": [20]},
    "bert": {"input_len": [64, 128, 256], "batch_size": [128], "epochs": [20],
             "num_bins": [20]}
}
