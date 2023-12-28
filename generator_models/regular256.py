from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import UpSampling2D, Conv2D


# image size is 256
def return_generator_model_and_imgsize(latent_dim=1024):
    # returns (Model, output image size)
    model = Sequential()

    model.add(Dense(24 * 16 * 16, activation="relu", input_dim=latent_dim))
    model.add(Reshape((16, 16, 24)))

    model.add(UpSampling2D())

    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())

    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())

    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    model.add(UpSampling2D())

    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("linear"))

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img), img.shape[1]

if __name__ == '__main__':
    M, i = return_generator_model_and_imgsize()
    M.summary()