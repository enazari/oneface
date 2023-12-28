import tensorflow as tf


def loss(y_true, y_pred):
    return tf.math.reduce_euclidean_norm(y_pred - y_true, axis=-1)


if __name__ == '__main__':
    import numpy as np
    import tensorflow as tf
    y_true = np.array([1,2,3.])
    y_pred = np.array([1,1,1.])
    print(loss(y_true, y_pred))
    print(-tf.math.reduce_euclidean_norm(y_pred - y_true, axis=-1))