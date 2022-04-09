import numpy as np
from global_settings import full_dict


def analyze_ssestm(O_hat):
    """ Analyze O_hat estimated from the ssestm model
    :param: estimated O_hat
    """

    # "涨" and "跌" with large positive/negative sentiments
    # "涨" and "跌" with highest occurrence
    sentiment = 0.5 * (O_hat[:, 0] - O_hat[:, 1])
    occurrence = 0.5 * (O_hat[:, 0] + O_hat[:, 1])

    sentiment_idx = np.argsort(sentiment)
    sentiment_sorted = np.array(full_dict)[sentiment_idx]

    occurrence_idx = np.argsort(occurrence)
    occurrence_sorted = np.array(full_dict)[occurrence_idx]

    return sentiment_sorted, occurrence_sorted
