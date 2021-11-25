import numpy as np


def experiment(df_rich, word_sps, window_iter, params_iter, fit_func, pre_func):
    """ train ssestm model to get the estimated O_hat
    :param df_rich: enriched dataframe
    :param word_sps: sparse word count matrix
    :param window_iter: rolling window iterator
    :param params_iter: parameters iterator
    :param fit_func: parameters iterator
    :param pre_func: parameters iterator
    :return: equal weighted and value weighted returns
    """

    for [trddt_train, trddt_valid, trddt_test] in window_iter:
        best_cum_e = -np.inf
        best_cum_v = -np.inf
        best_params_e = dict()
        best_params_v = dict()
        best_model_e = None
        best_model_v = None

        for params in params_iter:
            train_idx = df_rich["date_0"].apply(lambda _: _ in trddt_train)
            df_rich_train = df_rich.loc[train_idx, :].reset_index(inplace=False, drop=True)
            word_sps_train = word_sps[train_idx, ]
            model = fit_func(df_rich_train, word_sps_train, params)

            valid_ret_e = np.empty(len(trddt_valid))
            valid_ret_v = np.empty(len(trddt_valid))
            for i, dt in enumerate(trddt_valid):
                valid_idx = df_rich["date_0"].apply(lambda _: _ == dt)
                df_rich_valid = df_rich.loc[valid_idx, :].reset_index(inplace=False, drop=True)
                word_sps_valid = word_sps[valid_idx, :]
                valid_ret_e[i] = pre_func(df_rich_valid, word_sps_valid, params, model, ev="e")
                valid_ret_v[i] = pre_func(df_rich_valid, word_sps_valid, params, model, ev="v")

            valid_cum_e = np.log(np.cumprod(valid_ret_e + 1))
            valid_cum_v = np.log(np.cumprod(valid_ret_v + 1))

            if valid_cum_e[-1] > best_cum_e:
                best_cum_e = valid_cum_e[-1]
                best_params_e = params
                best_model_e = model

            if valid_cum_v[-1] > best_cum_v:
                best_cum_v = valid_cum_v[-1]
                best_params_v = params
                best_model_v = model

        ret_e = np.empty(len(trddt_test))
        ret_v = np.empty(len(trddt_test))
        for i, dt in enumerate(trddt_test):
            test_idx = df_rich["date_0"].apply(lambda _: _ == dt)
            df_rich_test = df_rich.loc[test_idx, :].reset_index(inplace=False, drop=True)
            word_sps_test = word_sps[test_idx, :]
            ret_e[i] = pre_func(df_rich_test, word_sps_test, best_params_e, best_model_e, ev="e")
            ret_v[i] = pre_func(df_rich_test, word_sps_test, best_params_v, best_model_v, ev="v")

        return ret_e, ret_v
