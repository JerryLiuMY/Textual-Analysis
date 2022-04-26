from sklearn.linear_model import LogisticRegression
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.utils import to_categorical
from datetime import datetime
import numpy as np


def fit_classifier(emb_vec, target, params):
    """ train classifier given emb_vec & sentiment
    :param emb_vec: embedding vector
    :param target: target
    :param params: parameters for the classifier
    """

    cls_type = params["cls_type"]
    if cls_type == "lr":
        cls_name = "Sklearn LR"
        cls = LogisticRegression(multi_class="ovr", n_jobs=1)
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {cls_name} Fitting classifier...")
        cls.fit(emb_vec, target)
    elif cls_type == "mlp":
        cls_name = "Keras DNN"
        hidden, num_bins = 75, len(target)
        cls = Sequential()
        cls.add(Input(shape=(emb_vec.shape[1], )))
        cls.add(Dense(hidden, kernel_regularizer=L2(l2=1e-2), activation="relu"))
        cls.add(Dense(hidden, kernel_regularizer=L2(l2=1e-2), activation="relu"))
        cls.add(Dense(hidden, kernel_regularizer=L2(l2=1e-2), activation="relu"))
        cls.add(Dense(num_bins, activation="softmax"))
        cls.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["accuracy"])
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {cls_name} Fitting classifier...")
        cls.fit(emb_vec, to_categorical(target, num_classes=num_bins), epochs=100, batch_size=512, verbose=2)
    else:
        raise ValueError("Invalid classifier type")

    return cls


def pre_classifier(emb_vec, cls, params):
    """ make prediction with classifier given emb_vec
    :param emb_vec: embedding vector
    :param cls: trained classifier
    :param params: parameters for the classifier
    """

    cls_type = params["cls_type"]

    if cls_type == "lr":
        target = cls.predict(emb_vec)
    elif cls_type == "mlp":
        target = np.argmax(cls.predict(emb_vec), axis=1)
    else:
        raise ValueError("Invalid classifier type")

    return target
