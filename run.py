if __name__ == "__main__":
    from main import create_dirs
    from global_settings import CLEAN_PATH
    from global_settings import RICH_PATH
    from global_settings import LOG_PATH
    PATHS = [CLEAN_PATH, RICH_PATH, LOG_PATH]
    create_dirs(PATHS)


# if __name__ == "__main__":
#     from main import run_data_prep
#     from main import run_word_sps
#     run_data_prep()
#     run_word_sps()


# if __name__ == "__main__":
#     from main import run_experiment
#     from params.params import perc_ls
#     model_name = "ssestm"
#     run_experiment(model_name, perc_ls)


if __name__ == "__main__":
    from main import run_backtest
    model_name = "ssestm"
    run_backtest(model_name)
