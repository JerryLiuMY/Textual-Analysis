perc_ls = 0.05

window_dict = {
    "train_win": 12,
    "valid_win": 12 + 6,
    "test_win": 12 + 6 + 1
}

num_bin = 20
params_text = {
    "d2v_emb": {"window": [10], "vector_size": [20], "epochs": [10], "num_bin": [num_bin]}
}

params_model = {
    "ssestm": {"pen": [0.0]},
    "doc2vec": {"num_bin": [num_bin]},
    "bert": {},
}
