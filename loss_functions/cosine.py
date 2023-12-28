import tensorflow as tf

cosine_loss = tf.keras.losses.CosineSimilarity(axis=-1)


def loss(y_true, y_pred):
    return cosine_loss(y_pred, y_true)


if __name__ == '__main__':
    import numpy as np
    a = np.array([1,2,3.])
    b = np.array([1,1,1.])
    print(loss(a,b))