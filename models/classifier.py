from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from sklearn.preprocessing import OneHotEncoder
import numpy as np


def fit_classifier(emb_vec, target, params):
    """ train classifier given emb_vec & sentiment
    :param emb_vec: embedding vector
    :param target: target
    :param params: parameters for the classifier
    """

    num_bins, hidden = params["num_bins"], params["hidden"]

    # Feedforward DNN
    enc = OneHotEncoder(categories=[np.arange(1, num_bins + 1)], sparse=False)
    target_enc = enc.fit_transform(target.reshape(-1, 1))

    cls = Sequential()
    cls.add(Input(shape=(emb_vec.shape[1], )))
    cls.add(Dense(hidden, kernel_regularizer=L2(1e-2), activation="relu"))
    cls.add(Dense(hidden, kernel_regularizer=L2(1e-2), activation="relu"))
    cls.add(Dense(num_bins, activation="softmax"))
    cls.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])
    cls.fit(emb_vec, target_enc, epochs=100, batch_size=512, verbose=2)

    return enc, cls
