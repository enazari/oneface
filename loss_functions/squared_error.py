import tensorflow.keras.backend as K


def loss(y_true, y_pred):
    return K.square(y_pred - y_true)
