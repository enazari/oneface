from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import UpSampling2D, Conv2D


# image size is 224
def return_generator_model_and_imgsize(latent_dim=224):
    # returns (Model, output image size)
    model = Sequential()

    model.add(Dense(1000 * 2 * 2, activation="relu", input_dim=latent_dim))
    model.add(Reshape((2, 2, 1000)))

    model.add(UpSampling2D())

    model.add(Conv2D(300, kernel_size=3, padding="same", activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling2D())

    model.add(Conv2D(300, kernel_size=3, padding="same", activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling2D())

    model.add(Conv2D(300, kernel_size=3, padding="same", activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling2D())

    model.add(Conv2D(300, kernel_size=3, padding="same", activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling2D())

    model.add(Conv2D(300, kernel_size=3, padding="same", activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling2D())

    model.add(Conv2D(300, kernel_size=3, padding="same", activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling2D())

    model.add(Conv2D(3, kernel_size=33))
    model.add(Activation("linear"))

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img), img.shape[1]

if __name__ == '__main__':
    M, i = return_generator_model_and_imgsize()
    M.summary()