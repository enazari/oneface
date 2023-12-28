from tensorflow.keras.layers import Input, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import UpSampling2D, Conv2D


# image size is 256
def return_generator_model_and_imgsize(latent_dim=512,return_raw_model=False):
    # returns (Model, output image size)
    model = Sequential()
    t = 3
    model.add(Dense(24 * t * t, activation="relu", input_dim=latent_dim))
    model.add(Reshape((t, t, 24)))

    model.add(Conv2DTranspose(100, kernel_size=10, strides=(2, 2), activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2DTranspose(100, kernel_size=15, strides=(2, 2), activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2DTranspose(100, kernel_size=20, strides=(2, 2), activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2DTranspose(3, kernel_size=26, strides=(2, 2), activation='linear'))
    model.add(BatchNormalization(momentum=0.8))

    model.summary()
    if return_raw_model: return model

    noise = Input(shape=(latent_dim,))
    img = model(noise)
    return Model(noise, img), img.shape[1]

if __name__ == '__main__':
    M, i = return_generator_model_and_imgsize()
    M.summary()