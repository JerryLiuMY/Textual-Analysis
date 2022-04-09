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

    return sentiment, occurrence
