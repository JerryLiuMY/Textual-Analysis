window_dict = {
    "train_win": 12,
    "valid_win": 12 + 6,
    "test_win": 12 + 6 + 1
}

params_dict = {
    "ssestm": {"pen": [0.0]},
    "doc2vec": {"window": [10], "vector_size": [20], "epochs": [10]}
}

perc_ls = 0.05
