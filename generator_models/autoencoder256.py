from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.layers import BatchNormalization, Activation
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import UpSampling2D, Conv2D


# image size is 256
def return_generator_model_and_imgsize(return_raw_model=False):
    # returns (Model, output image size)
    img_shape = (256, 256, 3)

    model = Sequential()

    model.add(Conv2D(128, kernel_size=10, padding="same", input_shape=img_shape))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # model.add(UpSampling2D())

    model.add(Conv2D(128, kernel_size=15, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # model.add(UpSampling2D())

    model.add(Conv2D(128, kernel_size=10, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))

    # model.add(UpSampling2D())

    model.add(Conv2D(3, kernel_size=3, padding="same"))
    model.add(Activation("linear"))

    model.summary()

    noise = Input(shape=img_shape)
    img = model(noise)

    if return_raw_model: return model
    return Model(noise, img), img.shape[1]

if __name__ == '__main__':
    M, i = return_generator_model_and_imgsize()
    M.summary()