import tensorflow as tf


def input_generated_imgs_output_unnormalized_imgs(gen_imgs):
    gen_imgs +=1
    gen_imgs *=127.5
    return gen_imgs

def unnormalizer(imgs):
    return input_generated_imgs_output_unnormalized_imgs(imgs)

def return_model(img_size, pooling=None):
    m =  tf.keras.applications.MobileNetV3Large(
        input_shape=(img_size, img_size, 3),
        alpha=1.0,
        minimalistic=False,
        include_top=False,
        weights="imagenet",
        input_tensor=None,
        classes=1000,
        pooling=pooling,
        dropout_rate=0.2,
        classifier_activation="softmax",
        include_preprocessing=False,
    )
    input = tf.keras.Input(shape= (img_size, img_size, 3) )
    x = m(input)
    reshaped = tf.keras.layers.Reshape((1280,))(x)

    return tf.keras.Model(inputs=input, outputs=reshaped)


def normalizer(imgs):
    imgs /= 127.7
    imgs -= 1
    return imgs

def return_normalizer():
    return normalizer


def return_model_and_unnormalizer(img_size, pooling='avg'):
    vgg16 = return_model(img_size, pooling)
    return vgg16, input_generated_imgs_output_unnormalized_imgs


if __name__ == '__main__':
    print( return_model(224).output_shape ) #(None, 1280)