from tensorflow.keras.layers import Input, Dense, Reshape, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import UpSampling2D, Conv2D


# image size is 256
def return_generator_model_and_imgsize(latent_dim=512):
    # returns (Model, output image size)
    model = Sequential()
    t = 5
    q = 4
    s = (2,2)
    f = 250
    model.add(Dense(24 * t * t, activation="relu", input_dim=latent_dim))
    model.add(Reshape((t, t, 24)))

    model.add(Conv2DTranspose(f, kernel_size=q, strides=s, activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2DTranspose(f, kernel_size=q, strides=s, activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2DTranspose(f, kernel_size=q, strides=s, activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2DTranspose(f, kernel_size=q, strides=s, activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2DTranspose(f, kernel_size=q, strides=s, activation='relu'))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Conv2DTranspose(3, kernel_size=3, strides=(1,1), activation='linear'))

    model.summary()
    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img), img.shape[1]

if __name__ == '__main__':
    M, i = return_generator_model_and_imgsize()
    M.summary()