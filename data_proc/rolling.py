from global_settings import trddt_all


def get_rolling(window_dict, date0_min, date0_max):
    trddt = trddt_all[(trddt_all >= date0_min) & (trddt_all <= date0_max)].tolist()
    trdmt = sorted(set([_[:-3] for _ in trddt]))
    trddt_chunck = [[d for d in trddt if d[:-3] == m] for m in trdmt]

    train_win = window_dict["train_win"]
    valid_win = window_dict["valid_win"]
    test_win = window_dict["test_win"]

    def flatten(chunck): return [item for sub_list in chunck for item in sub_list]

    for i in range(len(trddt_chunck) - test_win + 1):
        trddt_train_chunck = trddt_chunck[i: i + train_win]
        trddt_valid_chunck = trddt_chunck[i + train_win: i + valid_win]
        trddt_test_chunck = trddt_chunck[i + valid_win: i + test_win]
        trddt_train = flatten(trddt_train_chunck)
        trddt_valid = flatten(trddt_valid_chunck)
        trddt_test = flatten(trddt_test_chunck)

        yield [trddt_train, trddt_valid, trddt_test]
