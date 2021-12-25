# if __name__ == "__main__":
#     import os
#     from global_settings import CLEAN_PATH
#     from global_settings import RICH_PATH
#     from global_settings import LOG_PATH
#     PATHS = [CLEAN_PATH, RICH_PATH, LOG_PATH]
#
#     for path in PATHS:
#         if not os.path.isdir(path):
#             os.mkdir(path)


# if __name__ == "__main__":
#     from main import run_data_prep
#     run_data_prep()


# if __name__ == "__main__":
#     from main import run_textual
#     run_textual("word_sps")
#     run_textual("art_cut")


if __name__ == "__main__":
    from main import run_experiment
    from params.params import perc_ls
    model_name = "doc2vec"
    run_experiment(model_name, perc_ls)
