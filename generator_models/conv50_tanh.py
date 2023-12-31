from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import UpSampling2D, Conv2D


# image size is 224
def return_generator_model_and_imgsize(latent_dim=224):
    # returns (Model, output image size)
    model = Sequential()

    model.add(Dense(47 * 4 * 4, activation="relu", input_dim=latent_dim))
    model.add(Reshape((4, 4, 47)))

    model.add(UpSampling2D())

    model.add(Conv2D(300, kernel_size=3, padding="same", activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling2D())

    model.add(Conv2D(300, kernel_size=3, activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling2D())

    model.add(Conv2D(300, kernel_size=4, activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(UpSampling2D())

    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img), img.shape[1]

if __name__ == '__main__':
    M, i = return_generator_model_and_imgsize()
    M.summary()