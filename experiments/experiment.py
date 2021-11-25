

def evaluation(df_rich, word_sps, fit_func, pre_func, window_gen, params_gen):
    for [trddt_train, trddt_valid, trddt_test] in window_gen:
        for params in params_gen:
            train_idx = df_rich["date_0"].apply(lambda _: _ in trddt_train)
            df_rich_train = df_rich.loc[train_idx, :].reset_index(inplace=False, drop=True)
            word_sps_train = word_sps[train_idx, ]
            model = fit_func(df_rich_train, word_sps_train, params)

            for t in trddt_valid:
                valid_idx = df_rich["date_0"].apply(lambda _: _ == t)
                df_rich_valid = df_rich.loc[valid_idx, :].reset_index(inplace=False, drop=True)
                word_sps_valid = word_sps[valid_idx, ]
                ret = pre_func(df_rich_valid, word_sps_valid, params, model)
                ret_le, ret_se, ret_lv, ret_sv = ret

            for t in trddt_test:
                test_idx = df_rich["date_0"].apply(lambda _: _ in trddt_test)
                df_rich_test = df_rich.loc[test_idx, :].reset_index(inplace=False, drop=True)
                word_sps_test = word_sps[test_idx, ]
                ret = pre_func(df_rich_test, word_sps_test, params, model)
                ret_le, ret_se, ret_lv, ret_sv = ret

            pass
