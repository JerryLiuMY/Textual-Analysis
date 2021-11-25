import numpy as np
import datetime


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

    ret_e = np.empty(0)
    ret_v = np.empty(0)

    for [trddt_train, trddt_valid, trddt_test] in window_iter:
        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f"Training {trddt_train[0][:-3]} to {trddt_train[-1][:-3]}; "
              f"Validation {trddt_valid[0][:-3]} to {trddt_valid[-1][:-3]}; "
              f"Testing {trddt_test[0][:-3]} to {trddt_test[-1][:-3]} ")

        best_cum_e_scl = -np.inf
        best_cum_v_scl = -np.inf
        best_params_e = dict()
        best_params_v = dict()
        best_model_e = None
        best_model_v = None

        for params in params_iter:
            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                  f" Working on {'; '.join([str(key) + '=' +  str(value) for key, value in params.items()])}")

            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                  f"  Training...")
            train_idx = df_rich["date_0"].apply(lambda _: _ in trddt_train)
            df_rich_train = df_rich.loc[train_idx, :].reset_index(inplace=False, drop=True)
            word_sps_train = word_sps[train_idx, ]
            model = fit_func(df_rich_train, word_sps_train, params)

            print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
                  f"  Validation...")
            valid_ret_e_win = np.empty(len(trddt_valid))
            valid_ret_v_win = np.empty(len(trddt_valid))
            for i, dt in enumerate(trddt_valid):
                valid_idx = df_rich["date_0"].apply(lambda _: _ == dt)
                df_rich_valid = df_rich.loc[valid_idx, :].reset_index(inplace=False, drop=True)
                word_sps_valid = word_sps[valid_idx, :]
                valid_ret_e_win[i] = pre_func(df_rich_valid, word_sps_valid, params, model, ev="e")
                valid_ret_v_win[i] = pre_func(df_rich_valid, word_sps_valid, params, model, ev="v")

            valid_cum_e = np.log(np.cumprod(valid_ret_e_win + 1))
            valid_cum_v = np.log(np.cumprod(valid_ret_v_win + 1))

            if valid_cum_e[-1] > best_cum_e_scl:
                best_cum_e_scl = valid_cum_e[-1]
                best_params_e = params
                best_model_e = model

            if valid_cum_v[-1] > best_cum_v_scl:
                best_cum_v_scl = valid_cum_v[-1]
                best_params_v = params
                best_model_v = model

        print(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} "
              f" Building return...")
        ret_e_win = np.empty(len(trddt_test))
        ret_v_win = np.empty(len(trddt_test))
        for i, dt in enumerate(trddt_test):
            test_idx = df_rich["date_0"].apply(lambda _: _ == dt)
            df_rich_test = df_rich.loc[test_idx, :].reset_index(inplace=False, drop=True)
            word_sps_test = word_sps[test_idx, :]
            ret_e_win[i] = pre_func(df_rich_test, word_sps_test, best_params_e, best_model_e, ev="e")
            ret_v_win[i] = pre_func(df_rich_test, word_sps_test, best_params_v, best_model_v, ev="v")

        ret_e = np.concatenate([ret_e, ret_e_win], axis=0)
        ret_v = np.concatenate([ret_v, ret_v_win], axis=0)

        return ret_e, ret_v
