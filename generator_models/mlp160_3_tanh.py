from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import UpSampling2D, Conv2D


# image size is 256
def return_generator_model_and_imgsize(latent_dim=224):
    # returns (Model, output image size)
    image_dimention = 160
    model = Sequential()

    model.add(Dense(200, input_dim=latent_dim, activation='relu'))

    model.add(Dense(200, activation='relu'))

    model.add(Dense(200, activation='relu'))

    model.add(Dense(200, activation='relu'))

    model.add(Dense(200, activation='relu'))

    model.add(Dense(image_dimention * image_dimention * 3, activation='tanh'))

    model.add(Reshape((image_dimention, image_dimention, 3)))

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img), img.shape[1]


if __name__ == '__main__':
    M, i = return_generator_model_and_imgsize()
    M.summary()
