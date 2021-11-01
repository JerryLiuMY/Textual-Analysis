from global_setting import NEWS_PATH_LIST
import pandas as pd
import numpy as np
from multiprocessing import Pool
from old_files.word_filter import word_filter
# TODO: add to root words


def build_word_occur_return(news_df, dictionary):
    # Initialize word_occur_matrix
    permno_list = np.array(news_df['permno'])
    return_df = news_df[['est_date', 'est_time',
                         'open_price', 'open_cap', 'open_sp500', 'open_price_return', 'open_sp500_return',
                         'close_price', 'close_cap', 'close_sp500', 'close_price_return', 'close_sp500_return']]

    num_news = np.shape(news_df)[0]
    dict_size = len(dictionary)
    word_occur_matrix = np.zeros((num_news, dict_size))

    # Build word_occur_matrix
    for i, permno in enumerate(permno_list):
        news_article_list = news_df.loc[i, 'Article']
        news_title_list = news_df.loc[i, 'Title']
        news_word_list = news_article_list + news_title_list
        num_word = len(news_word_list)
        dict_count_list = dict_count(news_word_list, dictionary)

        for j, word in enumerate(dict_count_list):
            word_occur_matrix[i, j] = dict_count_list[j] / num_word

    # Build word_occur_ret_df
    word_occur_df = pd.DataFrame(word_occur_matrix, columns=dictionary)
    word_occur_return_df = pd.concat([word_occur_df, return_df], axis=1)
    word_occur_return_df['Permno'] = permno_list
    word_occur_return_df.set_index('Permno', inplace=True)

    return word_occur_return_df


def dict_count(news_word_list, dictionary):
    # Remove Negator
    negator = ['no', 'not', 'never']
    news_remove_list = []
    for id, word_raw in enumerate(news_word_list):
        if word_raw in negator:
            news_remove_list.append(id - 3)
            news_remove_list.append(id - 2)
            news_remove_list.append(id - 1)
            news_remove_list.append(id)
            news_remove_list.append(id + 1)
            news_remove_list.append(id + 2)
            news_remove_list.append(id + 3)
    news_remove_list = list(set(news_remove_list))

    # Create Dictionary Count List
    dict_count_list = np.zeros(len(dictionary), dtype='int')
    for id, word in enumerate(news_word_list):
        root_word = word_filter(word)
        if (id not in news_remove_list) and (root_word in dictionary):
            index = dictionary.index(root_word)
            dict_count_list[index] += 1

    return dict_count_list


if __name__ == '__main__':
    pool = Pool(4)
    pool.map(build_word_occur_return, NEWS_PATH_LIST)
