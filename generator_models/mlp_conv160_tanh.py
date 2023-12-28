from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import UpSampling2D, Conv2D


# image size is 256
def return_generator_model_and_imgsize(latent_dim=512):
    # returns (Model, output image size)
    image_dimention = 160
    model = Sequential()

    model.add(Dense(100, input_dim=latent_dim, activation='relu'))

    model.add(Dense(100, activation='relu'))

    model.add(Dense(160*160*1, activation='relu'))

    model.add(Reshape((160, 160, 1)))

    model.add(Conv2D(300, kernel_size=3, padding='same', activation='relu'))

    model.add(Conv2D(300, kernel_size=3, padding='same', activation='relu'))

    model.add(Conv2D(3, kernel_size=3, padding='same',  activation='tanh'))


    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img), img.shape[1]


if __name__ == '__main__':
    M, i = return_generator_model_and_imgsize()
    M.summary()
