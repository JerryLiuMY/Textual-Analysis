import numpy as np
from global_settings import full_dict


def ssestm_sentiment(O_hat):
    """ parameters
    :param: estimated O_hat
    """

    # "涨" and "跌" with large positive/negative sentiments
    # "涨" and "跌" with high occurrence
    sentiment = 0.5 * (O_hat[:, 0] - O_hat[:, 1])
    occurrence = 0.5 * (O_hat[:, 0] + O_hat[:, 1])
    idx = np.argsort(sentiment)

    rank_dict = np.array(full_dict)[idx]
    sentiment = sentiment[idx]
    occurrence = occurrence[idx]

    return rank_dict, sentiment, occurrence

