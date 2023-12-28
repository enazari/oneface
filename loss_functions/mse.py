import tensorflow as tf

mse = tf.keras.losses.MeanSquaredError()

def loss(y_true, y_pred):
    return mse(y_true, y_pred)


if __name__ == '__main__':
    import numpy as np
    a = np.array([1,2,4])
    b = np.array([1,1,1])
    print(loss(a,b))