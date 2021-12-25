date0_min = "2015-01-07"
date0_max = "2019-07-30"
perc_ls = 0.05

window_dict = {
    "train_win": 12,
    "valid_win": 12 + 6,
    "test_win": 12 + 6 + 1
}

params_dict = {
    "ssestm": {"pen": [0.0]},
    "doc2vec": {"window": [10], "vec_size": [20], "epochs": [15], "num_bins": [20]},
    "bert": {},
}
