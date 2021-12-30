from sklearn.linear_model import LogisticRegression
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import L2
from tensorflow.keras.utils import to_categorical


def fit_classifier(emb_vec, target, params):
    """ train classifier given emb_vec & sentiment
    :param emb_vec: embedding vector
    :param target: target
    :param params: parameters for the classifier
    """

    cls_type = params["cls_type"]

    if cls_type == "lr":
        cls = LogisticRegression(multi_class="ovr", n_jobs=1)
        cls.fit(emb_vec, target)
    elif cls_type == "mlp":
        num_bins, hidden = params["num_bins"], params["hidden"]
        cls = Sequential()
        cls.add(Input(shape=(emb_vec.shape[1], )))
        cls.add(Dense(hidden, kernel_regularizer=L2(l2=1e-2), activation="relu"))
        cls.add(Dense(hidden, kernel_regularizer=L2(l2=1e-2), activation="relu"))
        cls.add(Dense(hidden, kernel_regularizer=L2(l2=1e-2), activation="relu"))
        cls.add(Dense(num_bins, activation="softmax"))
        cls.compile(optimizer=Adam(), loss=CategoricalCrossentropy(), metrics=["accuracy"])
        cls.fit(emb_vec, to_categorical(target, num_classes=num_bins), epochs=100, batch_size=512, verbose=2)
    else:
        raise ValueError("Invalid classifier type")

    return cls
